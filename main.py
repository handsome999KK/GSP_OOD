
import argparse
from collections import OrderedDict
import math
import wandb
import time

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from data.dataset_3d import *
from utils.utils import get_dataset
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from data.dataset_3d import customized_collate_fn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP training and evaluation', add_help=False)
    # Data
    parser.add_argument("--dataset_name", type=str,
                        choices=["ScanObjectNN15", "ShapeNetCore54", "S3DIS7"],
                        help="Name of the dataset to use")
    parser.add_argument("--dataset_split", type=str,
                        choices=["SR1", "SR2", "SR3", "SN1", "SN2", "SN3"],
                        help="Name of the dataset to split")
    parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='shapenet', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--use_height', action='store_true', help='whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    # Model
    parser.add_argument('--model', default='ULIP_PN_SSG', type=str)
    # Training
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate_3d', action='store_true', help='eval 3d only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--test_ckpt_addr', default='', help='the ckpt to test 3d zero shot')
    return parser

best_acc1 = 0

def main(args):
    utils.init_distributed_mode(args)

    global best_acc1

    if utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='ULIP', id=wandb_id, config=args, reinit=True, entity='lxue')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.evaluate_3d:
        zero_stats = test_zeroshot_3d(args)
        print(zero_stats)
        return

    # create model
    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(args=args)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200, find_unused_parameters=False)

    # define loss function (criterion) and optimizer
    criterion = models.get_loss(args).cuda(args.gpu)

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            print('in optimizer freeze {}'.format(n))
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1']
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from the latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = get_dataset(train_transform, tokenizer, args, 'train')
    val_dataset = get_dataset(None, tokenizer, args, 'val')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        collate_fn=customized_collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    print(args)

    print("=> beginning training")

    best_epoch = -1

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)
        val_stats = {"acc1": -1}

        if epoch % 1 == 0:

            val_stats = test_zeroshot_3d_core(val_loader, model, tokenizer, args)
            acc1 = val_stats["acc1"]
            print(val_stats)

            is_best = acc1 > best_acc1
            if is_best:
                best_epoch = epoch

            best_acc1 = max(acc1, best_acc1)

            if is_best or epoch % 50 == 0:
                print("=> saving checkpoint")
                utils.save_on_master({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_acc1': best_acc1,
                        'args': args,
                    }, is_best, args.output_dir)

            if epoch + 1 == args.epochs:
                print("=> saving last checkpoint")
                utils.save_on_master({
                    'epoch': 'last',
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_acc1': best_acc1,
                    'args': args,
                }, is_best, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'best_acc1': best_acc1,
                     'best_epoch': best_epoch}

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
                # wandb.watch(model)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]

        pc = inputs[3]
        texts = inputs[2]

        image = inputs[4]
        inputs = [pc, texts, image]

        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale to [0, 100]

        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                        'scaler': scaler.get_scale(),
                        'logit': logit_scale})
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def test_zeroshot_3d_core(test_loader, model, tokenizer, args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    print('=> encoding captions')
    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    if args.dataset_name == "ScanObjectNN15":
        validate_dataset_name = args.dataset_name
        with open(os.path.join("./data/SR", 'labels.json')) as f:
            labels = json.load(f)[validate_dataset_name]
    elif args.dataset_name == "S3DIS7":
        validate_dataset_name = args.dataset_name
        with open(os.path.join("./data/S3D", 'labels.json')) as f:
            labels = json.load(f)[validate_dataset_name]
    else:
        print(f"error dataset choice")
        sys.exit(1)


    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = utils.get_model(model).encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            X = class_embeddings.cpu().numpy()
            KMEAN = 3
            kmeans = KMeans(n_clusters=KMEAN, random_state=42)
            kmeans.fit(X)

            labels = kmeans.labels_
            average_vectors = []
            for i in range(KMEAN):
                cluster_points = X[labels == i]
                average_vector = cluster_points.mean(axis=0) if len(cluster_points) > 0 else None
                average_vectors.append(average_vector)

            # 将平均向量转换为 tensor
            average_vectors_tensor = torch.tensor(average_vectors)
            class_embeddings = average_vectors_tensor
            for i in range(class_embeddings.size(0)):
                text_features.append(class_embeddings[i])
        text_features = torch.stack(text_features, dim=0)
        text_features = text_features.t()
        text_features = text_features.cpu().numpy()
        true_labels = []
        matrix_list = []
        for i, (pc, target) in enumerate(test_loader):
            pc = pc.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # encode pc
            pc_features = utils.get_model(model).encode_pc(pc)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            pc_features = pc_features.cpu().numpy()
            true_labels.extend(target.tolist())
            matrix_list.append(pc_features)
    progress.synchronize()

    all_pc_features = np.concatenate(matrix_list, axis=0)
    if args.dataset_name == "ScanObjectNN15":
        spilt = 5 * KMEAN
        spilt = int(spilt)
        spilt2 = 10 * KMEAN
        spilt2 = int(spilt2)
        if args.dataset_split == "SR1":
            text_features = text_features[:, :spilt]  # SR1
            true = [0 if label > 4 else 1 for label in true_labels]  # SR1
        elif args.dataset_split == "SR2":
            text_features = text_features[:, spilt:  spilt2]   #SR2
            true = [1 if 4 < num < 10 else 0 for num in true_labels]  # SR2
        elif args.dataset_split == "SR3":
            text_features = text_features[:, -spilt:]  # SR3
            true = [1 if label > 9 else 0 for label in true_labels]   #SR3
        else:
            print(f"error dataset split choice")
            sys.exit(1)

    if args.dataset_name == "S3DIS7":
        true = true_labels
    text_features = text_features.transpose()
    pos = text_features.shape[0]
    neg = 0
    max_positions = []
    min_positions = []
    for i in range(2):
        N = all_pc_features.shape[0]
        Adajency_Matrix_tensor = build_graph(text_features, all_pc_features)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Y = [1] * pos + [-1] * neg + [0] * N
        Y_array = torch.tensor(Y, dtype=torch.float32, device=device)
        Y_array_0 = Y_array.clone()
        a = 0.5
        n = 5
        for _ in range(n):
            Y_array = a * (Adajency_Matrix_tensor @ Y_array) + (1 - a) * Y_array_0
            Y_array[:pos + neg] = Y_array_0[:pos + neg]
        score_tensor = Y_array[-N:]
        score = score_tensor.cpu().numpy()
        pos, neg, text_features = self_train(score, max_positions, min_positions, all_pc_features, text_features, neg, pos, N)
    auroc = roc_auc_score(true, score)
    print(f'AUROC 值为: {auroc}')
    fpr, tpr, thresholds = roc_curve(true, score)
    closest_tpr_index = np.argmax(tpr >= 0.95)
    fpr95 = fpr[closest_tpr_index]
    print(f"FPR95: {fpr95:.4f}")

def self_train(score, max_positions, min_positions, all_pc_features, text_features, neg, pos, N, reserve = 5):
    score_max = score.copy()
    score_min = score.copy()
    for index in max_positions:
        if index < len(score):
            score_max[index] = 0
    for index in min_positions:
        if index < len(score):
            score_min[index] = 1
    max_indices = sorted(range(len(score_max)), key=lambda i: score_max[i], reverse=True)[:reserve]
    min_indices = sorted(range(len(score_min)), key=lambda i: score_min[i])[:reserve]

    max_position = [index for index in max_indices]
    min_position = [index for index in min_indices]
    max_positions.extend(max_position)
    min_positions.extend(min_position)
    max_pc = []
    min_pc = []
    for p in max_position:
        if p >= 0 and p < N:
            max_pc.append(all_pc_features[p])
    for p in min_position:
        if p >= 0 and p < N:
            min_pc.append(all_pc_features[p])
    min_pc = np.array(min_pc)
    num_duplicates = 0
    updated_rows = []
    for max_vector in max_pc:
        matched = False
        for a_vector in text_features:
            if np.array_equal(max_vector, a_vector):
                matched = True
                num_duplicates += 1
                break
        if not matched:
            updated_rows.append(max_vector)
    if updated_rows:
        text_features = np.vstack([np.array(updated_rows), text_features])
    pos = pos + reserve - num_duplicates
    num_duplicates = 0
    for min_vector in min_pc:
        matched = False
        for vector in text_features:
            if np.array_equal(min_vector, vector):
                matched = True
                num_duplicates += 1
                break
        if not matched:
            text_features = np.vstack([text_features, min_vector])
    neg = neg + reserve - num_duplicates
    return pos, neg, text_features

def build_graph(text_features, all_pc_features, KNN=10, index=10):
    K = text_features.shape[0]
    knn = NearestNeighbors(n_neighbors=KNN, metric='euclidean')
    knn.fit(all_pc_features)
    distances, indices = knn.kneighbors(all_pc_features)
    num_samples = all_pc_features.shape[0]
    sparse_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(KNN):
            neighbor_index = indices[i][j]
            euclidean_distance = np.linalg.norm(all_pc_features[i] - all_pc_features[neighbor_index])
            sparse_matrix[i][neighbor_index] = np.exp(-euclidean_distance)
    sparse_matrix_csr = csr_matrix(sparse_matrix)
    S = sparse_matrix_csr.toarray()
    distances = pairwise_distances(text_features, all_pc_features)
    nearest_indices = np.argsort(distances, axis=1)[:, :index]
    num_text_features = text_features.shape[0]
    exp_similarities_matrix = np.zeros((num_text_features, index))
    for i, indices in enumerate(nearest_indices):
        for j, idx in enumerate(indices):
            distance = np.linalg.norm(text_features[i] - all_pc_features[idx])
            exp_similarities_matrix[i, j] = np.exp(-distance)
    S_rank = S.shape[0]
    expanded_S = np.zeros((S_rank + K, S_rank + K))
    expanded_S[K:, K:] = S
    for i in range(exp_similarities_matrix.shape[0]):
        for j in range(exp_similarities_matrix.shape[1]):
            expanded_S[i, K + nearest_indices[i][j]] = exp_similarities_matrix[i, j]
    identity_matrix = np.eye(expanded_S.shape[0])
    expanded_S = np.where(identity_matrix == 1, 1, expanded_S)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    expanded_S_tensor = torch.from_numpy(expanded_S).float().to(device)
    W_tensor = expanded_S_tensor + expanded_S_tensor.t()
    d_tensor = expanded_S_tensor.sum(dim=1)
    d_tensor = 1.0 / torch.sqrt(d_tensor)
    D_neg_sqrt_tensor = torch.diag(d_tensor)
    Adajency_Matrix_tensor = D_neg_sqrt_tensor @ W_tensor @ D_neg_sqrt_tensor
    return Adajency_Matrix_tensor

def test_zeroshot_3d(args):
    #pdb.set_trace()
    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    try:
        model = getattr(models, old_args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    except:
        model = getattr(models, args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))

    tokenizer = SimpleTokenizer()

    test_dataset = get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )
    results = test_zeroshot_3d_core(test_loader, model, tokenizer, args)

    return results


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    moCUDA_VISIBLE_DEVICES = 0
    parser = argparse.ArgumentParser('ULIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
