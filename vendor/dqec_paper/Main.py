"""
Implementation of "Deep Quantum Error Correction" (DQEC), AAAI24
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
from __future__ import print_function
import argparse
import random
import os
import math
import csv
import json
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
from Codes import *
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import shutil
##################################################################
##################################################################

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)

##################################################################
class QECC_Dataset(data.Dataset):
    def __init__(self, code, ps, len, args):
        self.ps = ps
        self.len = len
        self.logic_matrix = code.logic_matrix.transpose(0, 1).clone().cpu()
        self.pc_matrix = code.pc_matrix.transpose(0, 1).clone().cpu()
        self.zero_cw = torch.zeros((self.pc_matrix.shape[0])).long()
        self.noise_method = self.independent_noise if args.noise_type == 'independent' else self.depolarization_noise
        self.noise_type = args.noise_type
        self.repetitions = args.repetitions
        
    def independent_noise(self,pp=None):
        pp = random.choice(self.ps) if pp is None else pp
        return np.random.binomial(1, pp, self.pc_matrix.shape[0])
    
    def depolarization_noise(self,pp=None):
        ## See original noise definition in https://github.com/Krastanov/neural-decoder/
        pp = random.choice(self.ps) if pp is None else pp
        out_dimZ = out_dimX = self.pc_matrix.shape[0]//2
        def makeflips(q):
            q = q/3.
            flips = np.zeros((out_dimZ+out_dimX,), dtype=np.dtype('b'))
            rand = np.random.rand(out_dimZ or out_dimX)
            both_flips  = (2*q<=rand) & (rand<3*q)
            ###
            x_flips = rand<  q
            flips[:out_dimZ] ^= x_flips
            flips[:out_dimZ] ^= both_flips
            ###
            z_flips = (q<=rand) & (rand<2*q)
            flips[out_dimZ:out_dimZ+out_dimX] ^= z_flips
            flips[out_dimZ:out_dimZ+out_dimX] ^= both_flips
            return flips
        flips = makeflips(pp)
        while not np.any(flips):
            flips = makeflips(pp)
        return flips*1.
        
        
    
    def __getitem__(self, index):
        x = self.zero_cw
        pp = random.choice(self.ps)
        if self.repetitions <= 1:
            z = torch.from_numpy(self.noise_method(pp))
            y = bin_to_sign(x) + z
            magnitude = torch.abs(y)
            syndrome = torch.matmul(z.long(),
                                    self.pc_matrix) % 2
            syndrome = bin_to_sign(syndrome) 
            return x.float(), z.float(), y.float(), (magnitude*0+1).float(), syndrome.float()
        ###
        ### See original setting definition in https://pymatching.readthedocs.io/en/stable/toric-code-example.html# 
        qq = pp
        
        noise_new = np.stack([self.noise_method(pp) for _ in range(self.repetitions)],1)
        noise_cumulative = (np.cumsum(noise_new, 1) % 2).astype(np.uint8)
        noise_total = noise_cumulative[:,-1]
        syndrome = (torch.matmul(torch.from_numpy(noise_cumulative).long().transpose(0,1),self.pc_matrix) % 2).transpose(0,1).numpy()
        syndrome_error = (np.random.rand(self.pc_matrix.shape[1], self.repetitions) < qq).astype(np.uint8)
        syndrome_error[:,-1] = 0 # Perfect measurements in last round to ensure even parity
        noisy_syndrome = (syndrome + syndrome_error) % 2
        # Convert to difference syndrome
        noisy_syndrome[:,1:] = (noisy_syndrome[:,1:] - noisy_syndrome[:,0:-1]) % 2
        
        z = torch.from_numpy(noise_total)
        syndrome = bin_to_sign(torch.from_numpy(noisy_syndrome)) #TODO: check if bin2sign is needed

        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        return x.float(), z.float(), (y*0+1).float(), (magnitude*0+1).float(), syndrome.float().transpose(0,1)
    
    def __len__(self):
        return self.len


##################################################################
##################################################################
class Binarization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        return grad_output*(torch.abs(x[0])<=1)

def binarization(y):
    return sign_to_bin(Binarization.apply(y))
###
def logical_flipped(L,x):
    return torch.matmul(x.float(),L.float()) % 2
###
def diff_GF2_mul(H,x):    
    H_bin = sign_to_bin(H) if -1 in H else H
    x_bin = x
    
    tmp = bin_to_sign(H_bin.unsqueeze(0)*x_bin.unsqueeze(-1))
    tmp = torch.prod(tmp,1)
    tmp = sign_to_bin(tmp)

    # assert torch.allclose(logical_flipped(H_bin,x_bin).cpu().detach().bool(), tmp.detach().cpu().bool())
    return tmp
##################################################################

def train(model, device, train_loader, optimizer, epoch, LR, max_batches=None):
    model.train()
    logic_matrix = train_loader.dataset.logic_matrix.to(device)
    cum_loss = cum_ber = cum_ler = cum_samples = 0
    cum_loss1 = cum_loss2 = cum_loss3 = 0
    t = time.time()
    # bin_fun = binarization
    bin_fun = torch.sigmoid
    for batch_idx, (x, z, y, magnitude, syndrome) in enumerate(train_loader):
        syndrome = syndrome.to(device)
        z_pred = model(magnitude.to(device), syndrome)
        loss1,loss2 = model.loss(-z_pred, z.to(device))
        loss3 = torch.nn.functional.binary_cross_entropy_with_logits((diff_GF2_mul(logic_matrix,bin_fun(-z_pred))),logical_flipped(logic_matrix, z.to(device)))
        loss = args.lambda_loss_ber*loss1+args.lambda_loss_n_pred*loss2+args.lambda_loss_ler*loss3
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ###
        z_pred = sign_to_bin(torch.sign(-z_pred))
        ber = BER(z_pred, z.to(device))
        ler = FER(logical_flipped(logic_matrix, z_pred), logical_flipped(logic_matrix, z.to(device)))

        cum_loss += loss.item() * z.shape[0]
        #
        cum_loss1 += loss1.item() * z.shape[0]
        cum_loss2 += loss2.item() * z.shape[0]
        cum_loss3 += loss3.item() * z.shape[0]
        #
        cum_ber += ber * z.shape[0]
        cum_ler += ler * z.shape[0]
        cum_samples += z.shape[0]
        #
        log_every = max(1, len(train_loader)//2)
        if (batch_idx+1) % log_every == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Vòng {epoch}, lô {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, loss={cum_loss / cum_samples:.5e}, BER={cum_ber / cum_samples:.3e}, LER={cum_ler / cum_samples:.3e}')
            logging.info(
                f'   thành phần loss: tổng={cum_loss / cum_samples:.5e}, logic={cum_loss3 / cum_samples:.5e}, lỗi vật lý={cum_loss1 / cum_samples:.5e}, ước lượng noise={cum_loss2 / cum_samples:.5e}')
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    logging.info(f'Thời gian vòng {epoch}: {time.time() - t:.1f} giây\n')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_ler / cum_samples


##################################################################

def test(model, device, test_loader_list, ps_range_test, cum_count_lim=10000):
    model.eval()
    test_loss_ber_list, test_loss_ler_list, cum_samples_all = [], [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ber = test_ler = cum_count = 0.
            logic_matrix = test_loader.dataset.logic_matrix.to(device)
            for x, z, y, magnitude, syndrome in test_loader:
                z_pred = model(magnitude.to(device), syndrome.to(device))
                _ = model.loss(-z_pred, z.to(device))
                z_pred = sign_to_bin(torch.sign(-z_pred))

                test_ber += BER(z_pred, z.to(device)) * z.shape[0]
                test_ler += FER(logical_flipped(logic_matrix, z_pred), logical_flipped(logic_matrix, z.to(device)) ) * z.shape[0]
                cum_count += z.shape[0]
                if cum_count > cum_count_lim:
                    break
            cum_samples_all.append(cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_ler_list.append(test_ler / cum_count)
            print(f'Đánh giá p={ps_range_test[ii]:.3e}: BER (tỷ lệ lỗi bit)={test_loss_ber_list[-1]:.3e}, LER (tỷ lệ lỗi logic)={test_loss_ler_list[-1]:.3e}')
        ###
        logging.info('LER (tỷ lệ lỗi logic): ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ler_list, ps_range_test))]))
        logging.info('BER (tỷ lệ lỗi bit): ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, ps_range_test))]))
        logging.info(f'Trung bình LER={np.mean(test_loss_ler_list):.3e}, BER={np.mean(test_loss_ber_list):.3e}')
    logging.info(f'Số mẫu đánh giá theo từng p: {cum_samples_all}; thời gian: {time.time() - t:.1f} giây\n')
    return test_loss_ber_list, test_loss_ler_list

##################################################################
##################################################################
##################################################################


def choose_device(requested):
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("Thiết bị MPS không khả dụng trong môi trường Python hiện tại.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA không khả dụng trong môi trường Python hiện tại.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cpu_copy(value):
    """Chuyển dữ liệu checkpoint về CPU để lưu và mở được trên máy khác."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step,
                    validation_mean_ler, args):
    """Lưu trọng số, trạng thái tối ưu hóa và thông tin lần chạy."""
    safe_args = {
        key: value for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, type(None), list, tuple))
    }
    payload = {
        "model_state_dict": cpu_copy(model.state_dict()),
        "optimizer_state_dict": cpu_copy(optimizer.state_dict()),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "validation_mean_ler": validation_mean_ler,
        "args": safe_args,
    }
    temporary_path = path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def run_fixed_validation(model, device, loaders, probabilities, sample_count, seed):
    """Dùng cùng một tập validation giữa các lần đánh giá, không đổi RNG train."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        return test(model, device, loaders, probabilities, cum_count_lim=sample_count)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def write_evaluation_csv(path, probabilities, bers, lers):
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["xac_suat_loi_p", "ty_le_loi_bit_BER", "ty_le_loi_logic_LER"])
        writer.writerows(zip(probabilities, bers, lers))


def main(args):
    device = choose_device(args.device)
    print(f"Thiết bị huấn luyện: {device}")
    args.code.logic_matrix = args.code.logic_matrix.to(device) 
    args.code.pc_matrix = args.code.pc_matrix.to(device) 
    code = args.code
    assert 0 < args.repetitions 
    if args.repetitions > 1:
        from Model_T_measurements import ECC_Transformer
    else:
        from Model import ECC_Transformer

    #################################
    model = ECC_Transformer(args, dropout=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    global_step = 0
    best_validation_ler = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        saved_args = checkpoint.get('args', {})
        required_match = (
            'steps', 'steps_per_epoch', 'batch_size', 'code_type', 'code_L',
            'noise_type', 'repetitions', 'N_dec', 'd_model', 'h', 'lr',
            'lambda_loss_ber', 'lambda_loss_ler', 'lambda_loss_n_pred',
            'no_g', 'no_mask',
        )
        mismatches = [
            name for name in required_match
            if name in saved_args and saved_args[name] != getattr(args, name)
        ]
        if mismatches:
            raise SystemExit(
                'Cấu hình resume không khớp checkpoint ở: ' + ', '.join(mismatches)
                + '. Hãy dùng lại cấu hình train ban đầu.'
            )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1
        global_step = int(checkpoint['global_step'])
        best_path = os.path.join(args.path, 'best_checkpoint.pt')
        if os.path.exists(best_path):
            best_checkpoint = torch.load(best_path, map_location='cpu', weights_only=True)
            best_validation_ler = float(best_checkpoint['validation_mean_ler'])
        logging.info(
            f"Đã tiếp tục từ checkpoint vòng {checkpoint['epoch']}, "
            f"bước {global_step}/{args.steps}; LER validation tốt nhất "
            f"hiện tại={best_validation_ler:.3e}."
        )
    
    logging.info(f'Ma trận kiểm tra có kích thước {code.pc_matrix.shape}')
    logging.info(
        f'Kiến trúc QECCT: {args.N_dec} tầng attention, chiều ẩn {args.d_model}, '
        f'{args.h} đầu attention.'
    )
    logging.info(f'Số tham số của mô hình: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    physical_label_count = code.pc_matrix.shape[1]
    logical_label_count = code.logic_matrix.shape[0]
    logging.info(
        f'Mỗi bản ghi được sinh khi chạy: {physical_label_count} bit lỗi vật lý; '
        f'{logical_label_count} bit nhãn logic suy ra từ lỗi đó.'
    )
    logging.info(
        f'Ngân sách: {args.steps:,} bước cập nhật × {args.batch_size} mẫu/bước '
        f'= tối đa {args.steps * args.batch_size:,} lượt lấy mẫu tổng hợp; '
        'dữ liệu không được lưu thành một tệp cố định.'
    )
    #################################
    ps_test = np.linspace(0.01, 0.2, 9)
    if args.noise_type == 'depolarization':
        ps_test = np.linspace(0.05, 0.2, 9)
    if args.repetitions > 1:
        ps_test = np.linspace(0.02, 0.04, 9)
    ###
    ps_train = ps_test

    train_dataloader = DataLoader(
        QECC_Dataset(code, ps_train, len=args.batch_size * args.steps_per_epoch, args=args),
        batch_size=int(args.batch_size), shuffle=True, num_workers=args.workers,
        persistent_workers=args.workers > 0)
    test_dataloader_list = [DataLoader(QECC_Dataset(code, [ps_test[ii]], len=args.eval_samples,args=args),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=0) for ii in range(len(ps_test))]
    #################################
    history_path = os.path.join(args.path, 'training_history.csv')
    if not args.resume or not os.path.exists(history_path):
        with open(history_path, 'w', newline='', encoding='utf-8') as history_file:
            csv.writer(history_file).writerow([
                'vong', 'buoc_cap_nhat', 'train_loss', 'train_BER', 'train_LER',
                'validation_mean_LER'
            ])
    for epoch in range(start_epoch, args.epochs + 1):
        steps_this_epoch = min(args.steps_per_epoch, args.steps - global_step)
        if steps_this_epoch <= 0:
            break
        loss, ber, ler = train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0], max_batches=steps_this_epoch)
        global_step += steps_this_epoch
        scheduler.step()
        save_checkpoint(os.path.join(args.path, 'last_checkpoint.pt'), model,
                        optimizer, scheduler, epoch, global_step, None, args)
        validation_mean_ler = ''
        if epoch % args.eval_every_epochs == 0 or epoch in [1, args.epochs]:
            _, validation_lers = run_fixed_validation(
                model, device, test_dataloader_list, ps_test, args.eval_samples,
                args.seed + 10000)
            validation_mean_ler = float(np.mean(validation_lers))
            logging.info(f'LER trung bình trên validation: {validation_mean_ler:.3e}')
            if validation_mean_ler < best_validation_ler:
                best_validation_ler = validation_mean_ler
                save_checkpoint(os.path.join(args.path, 'best_checkpoint.pt'), model,
                                optimizer, scheduler, epoch, global_step,
                                best_validation_ler, args)
                logging.info('Đã lưu checkpoint tốt nhất theo LER validation.')
        with open(history_path, 'a', newline='', encoding='utf-8') as history_file:
            csv.writer(history_file).writerow([
                epoch, global_step, loss, ber, ler, validation_mean_ler
            ])
    ###
    best_checkpoint = torch.load(os.path.join(args.path, 'best_checkpoint.pt'),
                                 map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    logging.info(f"Đã nạp checkpoint tốt nhất ở vòng {best_checkpoint['epoch']} "
                 f"(bước {best_checkpoint['global_step']}) để đánh giá cuối.")
    ps_test = np.linspace(0.01, 0.2, 18)
    if args.noise_type == 'depolarization':
        ps_test = np.linspace(0.05, 0.2, 18)
    if args.repetitions > 1:
        ps_test = np.linspace(0.02, 0.04, 18)
    ###
    test_dataloader_list = [DataLoader(QECC_Dataset(code, [ps_test[ii]], len=args.eval_samples,args=args),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=0) for ii in range(len(ps_test))]
    final_bers, final_lers = run_fixed_validation(
        model, device, test_dataloader_list, ps_test, args.eval_samples,
        args.seed + 20000)
    write_evaluation_csv(os.path.join(args.path, 'evaluation_final.csv'),
                         ps_test, final_bers, final_lers)
    logging.info('Đã ghi các chỉ số đánh giá cuối vào evaluation_final.csv')
##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Huấn luyện QECCT theo paper trên CPU, CUDA hoặc Apple MPS.')
    parser.add_argument('--steps', type=int, default=100000, help='tổng số batch cập nhật; mặc định 100.000 bước để bắt đầu')
    parser.add_argument('--steps-per-epoch', type=int, default=5000, help='số batch trong mỗi vòng')
    parser.add_argument('--epochs', type=int, default=None, help='nếu đặt, ghi đè ngân sách --steps bằng epochs × steps-per-epoch')
    parser.add_argument('--eval-every-epochs', type=int, default=5)
    parser.add_argument('--eval-samples', type=int, default=10000, help='số mẫu kiểm tra cho mỗi xác suất lỗi')
    parser.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--workers', type=int, default=4,
                        help='tiến trình CPU để sinh mẫu train; đặt 0 để tắt')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='thư mục kết quả; nếu bỏ trống sẽ tạo thư mục có dấu thời gian')
    parser.add_argument('--resume', type=str, default=None,
                        help='tiếp tục từ last_checkpoint.pt đã lưu')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='toric',choices=['toric'])
    parser.add_argument('--code_L', type=int, default=4,help='Lattice length')
    parser.add_argument('--repetitions', type=int, default=1,help='Number of faulty repetitions. <=1 is equivalent to none.')
    parser.add_argument('--noise_type', type=str,default='independent', choices=['independent','depolarization'],help='Noise model')

    # model args
    parser.add_argument('--N_dec', type=int, default=6,help='Number of QECCT self-attention modules')
    parser.add_argument('--d_model', type=int, default=128,help='QECCT dimension')
    parser.add_argument('--h', type=int, default=16,help='Number of heads')

    # qecc args
    parser.add_argument('--lambda_loss_ber', type=float, default=0.5,help='BER loss regularization')
    parser.add_argument('--lambda_loss_ler', type=float, default=1.,help='LER loss regularization')
    parser.add_argument('--lambda_loss_n_pred', type=float, default=0.5,help='g noise prediction regularization')
    
    # ablation args
    parser.add_argument('--no_g', type=int, default=0)
    parser.add_argument('--no_mask', type=int, default=0)

    args = parser.parse_args()
    if args.steps <= 0 or args.steps_per_epoch <= 0 or args.eval_samples <= 0 or args.workers < 0:
        parser.error('--steps, --steps-per-epoch và --eval-samples phải dương')
    if args.eval_every_epochs <= 0:
        parser.error('--eval-every-epochs phải dương')
    if args.epochs is not None:
        if args.epochs <= 0:
            parser.error('--epochs phải dương')
        args.steps = args.epochs * args.steps_per_epoch
    else:
        args.epochs = math.ceil(args.steps / args.steps_per_epoch)
    set_seed(args.seed)
    ####################################################################
    if args.no_g > 0:
        args.lambda_loss_n_pred= 0.
    ###
    class Code():
        pass
    code = Code()
    H,Lx = eval(f'Get_{args.code_type}_Code')(args.code_L,full_H=args.noise_type == 'depolarization')
    code.logic_matrix = torch.from_numpy(Lx).long()
    code.pc_matrix = torch.from_numpy(H).long()
    code.n = code.pc_matrix.shape[1]
    code.k = code.n - code.pc_matrix.shape[0]
    code.code_type = args.code_type
    args.code = code
    ####################################################################
    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if not os.path.isfile(resume_path):
            parser.error(f'không tìm thấy checkpoint: {resume_path}')
        resume_dir = os.path.dirname(resume_path)
        if args.output_dir and os.path.abspath(args.output_dir) != resume_dir:
            parser.error('--output-dir phải là thư mục chứa checkpoint khi dùng --resume')
        model_dir = resume_dir
    else:
        model_dir = args.output_dir or os.path.join(
            'Final_Results_QECCT', args.code_type,
            'Code_L_' + str(args.code_L),
            f'noise_model_{args.noise_type}',
            f'repetition_{args.repetitions}',
            datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Thư mục lưu model và log: {model_dir}")
    logging.info(
        f"Thiết lập: code={args.code_type}, L={args.code_L}, noise={args.noise_type}, "
        f"repetitions={args.repetitions}, steps={args.steps}, batch={args.batch_size}, "
        f"seed={args.seed}, worker CPU={args.workers}"
    )
    with open(os.path.join(model_dir, 'run_config.json'), 'w', encoding='utf-8') as config_file:
        json.dump({
            'device_requested': args.device,
            'device_used': str(choose_device(args.device)),
            'paper': 'Deep Quantum Error Correction (AAAI 2024), QECCT',
            'source': 'https://github.com/yoniLc/DQEC',
            'code': args.code_type,
            'code_L': args.code_L,
            'physical_error_bits_per_record': int(code.n),
            'logical_bits_per_record': int(code.logic_matrix.shape[0]),
            'noise_type': args.noise_type,
            'repetitions': args.repetitions,
            'optimizer_updates': args.steps,
            'batch_size': args.batch_size,
            'generated_training_samples_upper_bound': args.steps * args.batch_size,
            'data_storage': 'generated on the fly; not saved as a fixed dataset',
            'model_layers': args.N_dec,
            'model_dimension': args.d_model,
            'attention_heads': args.h,
            'seed': args.seed,
        }, config_file, ensure_ascii=False, indent=2)

    main(args)

