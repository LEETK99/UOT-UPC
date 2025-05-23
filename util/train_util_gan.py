import os
import time
import numpy as np
from tqdm import tqdm
from util import *
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from loss import *

torch.autograd.set_detect_anomaly(True)





os.system('clear')


class TrainStatus():
    def __init__(self, epoch_num, train_num, valid_num, train_log_num=10):
        self.train_log = []
        self.valid_log = []
        self.epoch_num = epoch_num
        self.train_num = train_num 
        self.valid_num = valid_num 
        self.train_log_num = train_log_num
        self.last_save_path = None

    def add_train_log(self, log):
        self.train_log.append(log)
        if len(self.train_log) > self.train_log_num:
            self.train_log.pop(0)

    def add_valid_log(self, log):
        self.valid_log = log

    def add_save_path(self, path):
        self.last_save_path = path

    def print(self, curr_epoch, curr_train, curr_valid, go_back=True):
        print('\n')
        text = '<EPOCH: %d/%d>' % (curr_epoch, self.epoch_num)
        text += '<TRAIN: %d/%d>' % (curr_train, self.train_num)
        text += '<VALID: %d/%d>' % (curr_valid, self.valid_num)
        clear_print_line(text)
        clear_print_line('LAST SAVE PATH: '+str(self.last_save_path))
        clear_print_line(('LAST %d TRAIN LOG:' % self.train_log_num) + (' None' if len(self.train_log)==0 else ''))
        num = 4
        for log in self.train_log:
            clear_print_line(log)
            num += 1
        clear_print_line('LAST VALID LOG:' + (' None' if len(self.valid_log)==0 else ''))
        num += 1
        for log in self.valid_log:
            clear_print_line(log)
            num += 1
        if go_back:
            move_up(num+1)
        return num+1


class TrainFramework():
    def __init__(self, args, cuda_index):
        self.batch_size = args.batch_size
        self.log_dir = f'../train_logs/{args.log_dir}/' + args.class_name
        self.is_restore = args.restore
        self.train_crush = True
        self.max_line = 10
        self.cuda_index = cuda_index
        self.betas_g = tuple(args.betas_g)
        self.betas_d = tuple(args.betas_d)
        self.clip = args.clip
        os.makedirs(self.log_dir, exist_ok=True)

        args2json(args, self.log_dir)

        self.logger = PrintLogger(self.log_dir, self.is_restore, prefix='[TRAIN]')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        
        if cuda_index is not None and str(self.device) != 'cuda':
            self.logger.log_warn('CUDA device %s is not available' % cuda_index)
        text = 'Device: ' + str(self.device)
        if str(self.device) == 'cuda':
            text += ' ' + cuda_index
        text += ', Batch size: %d' % self.batch_size
        self.logger.log_info(text)

    def _set_dataset(self, train_dataset_name, valid_dataset_name, train_dataset, valid_dataset):
        self.train_dataset_name = train_dataset_name
        self.valid_dataset_name = valid_dataset_name
        # self.logger.log_info('Train dataset name: ' + self.train_dataset_name)
        # self.logger.log_info('Valid dataset name: ' + self.valid_dataset_name)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_data_num = len(self.train_dataset)
        self.valid_data_num = len(self.valid_dataset)
        self.logger.log_info('Train dataset name: %s, len: %d' % (self.train_dataset_name, self.train_data_num))
        self.logger.log_info('Valid dataset name: %s, len: %d' % (self.valid_dataset_name, self.valid_data_num))
        
        self.num_work = 4
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_work, collate_fn=self.train_dataset._collate_fn)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_work, collate_fn=self.valid_dataset._collate_fn)
        self.train_step = len(self.train_loader)
        self.valid_step = len(self.valid_loader)
        self.train_iter = iter(self.train_loader)
        self.valid_iter = iter(self.valid_loader)
        
        self.logger.log_info('Work num: %d, Train num: %d, Train step: %d, Valid num: %d, Valid step: %d' % (self.num_work, self.train_data_num, self.train_step, self.valid_data_num, self.valid_step))

    def _set_net(self, net, net_name=''):
        self.net = net
        self.net.to(self.device)        
        self.logger.log_info('Net: ' + net_name)
        
    def _set_loss(self, train_loss, test_loss):
        self.loss = train_loss
        self.loss_test = test_loss

        loss_names = self.loss.loss_name
        loss_names_string = 'LOSS NAMES: '
        for name in loss_names:
            loss_names_string += (name+', ')
        loss_names_string = loss_names_string[:-2]
        self.logger.log_info(loss_names_string)
        loss_names = self.loss_test.loss_name
        loss_names_string = 'LOSS TEST NAMES: '
        for name in loss_names:
            loss_names_string += (name+', ')
        loss_names_string = loss_names_string[:-2]
        self.logger.log_info(loss_names_string)

    def _set_optimzer(self, optimizer, **kwargs):
        self.lr_g = kwargs['lr_g']
        self.lr_d = kwargs['lr_d']
        if 'weight_decay' in kwargs.keys():
            self.weight_decay = kwargs['weight_decay']
        else:
            self.weight_decay = 0
        
        self.G_freeze, self.D_freeze = False, False

        if self.net.typeG in ['T1', 'T2']:
            self.optimizer_G = torch.optim.Adam(self.net.G1.parameters(), lr=self.lr_g, betas=self.betas_g, weight_decay=self.weight_decay)
        else:
            self.optimizer_G = torch.optim.Adam(list(self.net.G1.parameters())+list(self.net.G2.parameters()), lr=self.lr_g, betas=self.betas_g, weight_decay=self.weight_decay)

        self.optimizer_D = torch.optim.Adam(self.net.D.parameters(), lr=self.lr_d, betas=self.betas_d, weight_decay=self.weight_decay)
        
        self.logger.log_info('Set G freeze: ' + str(self.G_freeze) + ', set D freeze: ' + str(self.D_freeze))
        self.logger.log_info('Set Adam optimizer, lr_g: %f, lr_d: %f, weight_decay: %f' % (self.lr_g, self.lr_d, self.weight_decay))
        
    def find_model_path(self, last_epoch):
        if last_epoch is not None:
            last_model_path = os.path.join(self.log_dir, 'model-%d.pkl' % last_epoch)
            if os.path.exists(last_model_path):
                return last_model_path, last_epoch + 1 
            else:
                return None, 1
        else:
            for root, dirs, files in os.walk(self.log_dir):
                break
            epoch = -1
            for f in files:
                s = f.find('-')
                e = f.find('.')
                if s != -1 and e != -1: 
                    try:
                        epoch = max(epoch, int(f[s+1:e]))
                    except:
                        pass
            if epoch == -1:
                return None, 1
            else:
                last_model_path = os.path.join(self.log_dir, 'model-%d.pkl' % epoch)
                return last_model_path, epoch + 1
    
    def train(self, epoch, G_opt_step=1, D_opt_step=1,\
        print_pre_step=100, test_pre_step=-1, save_pre_epoch=10,\
        last_epoch=None, after_restore_func=None, after_valid_func=None, on_save_func=None, restore_func=None,\
        epoch_test=True):
        '''
        after_valid_func(self, loss_recorder)       here, 'self' means training framework itself
        restore_func(self) --> last_model_path
        after_restore_func(self)
        on_save_func(self)

        epoch:  int the number of epoch of training
        epoch_test: do test after each epoch
        test_pre_step:  set to -1 if don't test according to step

        G_opt_step: update Generator after 'G_opt_step' step(s)
        D_opt_step: update Discriminator after 'G_opt_step' step(s)

        '''
        self.load_cnt = 0
        writer = SummaryWriter(self.log_dir)
        self.epoch = epoch
        self.print_pre_step = print_pre_step 
        self.test_pre_step = test_pre_step 
        self.save_pre_epoch = save_pre_epoch
        self.epoch_test = epoch_test
        if test_pre_step == -1:
            self.test_num_pre_epoch = 0
            self.test_pre_step = self.train_step + 10
        else:
            self.test_num_pre_epoch = self.train_step // test_pre_step
        if epoch_test:
            self.test_num_pre_epoch += 1
        # self.test_num_pre_epoch = self.train_step // test_pre_step
        start_epoch = 1
        if self.is_restore:
            if restore_func is not None:
                last_model_path = restore_func(self)
            else:
                last_model_path, start_epoch = self.find_model_path(last_epoch)
                if last_model_path is not None:
                    self.net.load_state_dict(torch.load(last_model_path))
                    self.logger.log_info('Restore last model: ' + last_model_path)
                else:
                    if last_epoch is None:
                        self.logger.log_warn('No last model pkl')
                    else:
                        self.logger.log_warn('No model-%d.pkl' % last_epoch)
            if after_restore_func is not None:
                after_restore_func(self)
        else:
            self.logger.log_info('New train')
        self.logger.log_info('Max epoch: %d. Trian start from epoch: %d, at time: ' % (self.epoch, start_epoch) + get_current_time())
        

        epoch -= start_epoch

        self.save_num = (epoch+1) // save_pre_epoch

        self.train_state = TrainStatus(epoch, self.train_step, self.test_num_pre_epoch)
        if self.is_restore:
            self.train_state.add_save_path(last_model_path)

        self.total_tqdm_num = int((epoch+1)*(self.train_step + self.test_num_pre_epoch*self.valid_step)+self.save_num)

       
        self.total_tqdm_num += 1
        t = tqdm(total=self.total_tqdm_num, ncols=100)
        t.update()
        self.train_state.print(start_epoch, 0, 0)
        for epoch in range(start_epoch, self.epoch+1):
            self.curr_epoch = epoch
            self.curr_valid = 0
            
            mean_loss = np.zeros([self.loss.loss_num]) 

            for i, data in enumerate(self.train_loader):
                self.curr_step = i + 1
                self.curr_train = self.curr_step
                
                data = self.train_dataset.to_device(data, self.device)
                # inputs, gts, info = data
                
                outputs = self.net(data) 
                samples, values = outputs
                loss = self.loss(samples, values)
                mean_loss += np.array([l.item() for l in loss])

                self.optimizer_G.zero_grad()
                loss[0].backward(retain_graph=True)

                self.optimizer_D.zero_grad()
                loss[1].backward() 

                if self.clip != 0:
                    nn.utils.clip_grad_norm_(list(self.net.G1.parameters())+list(self.net.G2.parameters()), self.clip)
                    nn.utils.clip_grad_norm_(self.net.D.parameters(), self.clip)
                
                self.optimizer_G.step()
                self.optimizer_D.step()

                t.update()

                if self.curr_step % print_pre_step == 0:
                    train_log = '[T]<Epoch: %d, Train: %d>{' % (self.curr_epoch, self.curr_train)
                    
                    mean_loss /= print_pre_step 
                    for idx, l in enumerate(mean_loss):
                        train_log += ('%f, ' % l)
                        writer.add_scalar(f"train: {self.loss.loss_name[idx]}", l, (epoch-1)*self.train_step + self.curr_step)
                    train_log = train_log[:-2]+'}'
                    self.train_state.add_train_log(train_log)
                    self.logger.log_file(train_log)
                    mean_loss = np.zeros([self.loss.loss_num]) 

            if self.epoch_test:
                loss_dict = {}
                self.net = self.net.eval()
                for j, data in enumerate(self.valid_loader):
                    data = self.valid_dataset.to_device(data, self.device)
                    # inputs, gts, info = data[0]
                    info = data[0][-1]
                    
                    with torch.no_grad():
                        outputs = self.net(data)
                        samples, _ = outputs
                        loss = self.loss_test.batch_forward(samples, data)

                    for bi, inf in enumerate(info):
                        if inf not in loss_dict:
                            loss_dict[inf] = [np.zeros([self.loss_test.loss_num]), 0]
                        loss_dict[inf][0] += np.array([l[bi].item() for l in loss])
                        loss_dict[inf][1] += 1
                    t.update()

                self.net = self.net.train()

                self.curr_valid += 1
                valid_log = []
                valid_log.append('[V]<Epoch: %d, Valid: %d>' % (self.curr_epoch, self.curr_valid))
                avg_loss = np.zeros([self.loss_test.loss_num])
                avg_num = 0
                loss_recorder = {}       # record loss for after_valid_func
                for obj_class in loss_dict:
                    log = obj_class + ': '
                    obj_loss = loss_dict[obj_class][0]
                    obj_num = loss_dict[obj_class][1]
                    avg_loss += obj_loss
                    avg_num += obj_num
                    obj_loss /= obj_num
                    for idx, l in enumerate(obj_loss):
                        log += ('%f, ' % l)
                        writer.add_scalar(f'valid(epoch):{self.loss_test.loss_name[idx]}:', l, epoch)
                    valid_log.append(log[:-2])
                    loss_recorder[obj_class] = obj_loss
                avg_loss /= avg_num
                log = 'AVG: '
                for l in avg_loss:
                    log += ('%f, ' % l)
                valid_log.append(log[:-2])
                loss_recorder['AVG'] = avg_loss
                self.train_state.add_valid_log(valid_log)
                self.logger.log_file(valid_log)
                if after_valid_func is not None:
                    after_valid_func(self, loss_recorder)
            if epoch % save_pre_epoch == 0:
                save_path = os.path.join(self.log_dir, 'model-%d.pkl' % epoch)
                torch.save(self.net.state_dict(), save_path)
                self.train_state.add_save_path(save_path)
                t.update()
                self.logger.log_file('Save to: ' + save_path)
                self.train_state.print(self.curr_epoch, self.curr_train, self.curr_valid)
            if on_save_func is not None:
                on_save_func(self)

        t.close()
        writer.close()
        self.train_state.print(self.curr_epoch, self.curr_train, self.curr_valid, go_back=False)
        
        self.train_crush = False
        

    def __del__(self):
        if self.train_crush:
            self.logger.log_warn('TRAIN CRUSH AT TIME: ' + get_current_time())
        else:
            self.logger.log_info('Train over at time: ' + get_current_time())
 



