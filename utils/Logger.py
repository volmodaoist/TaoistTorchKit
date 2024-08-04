import os
import logging


class Logger:
    def __init__(self, log_file, task_type='multiclass'):
        self.logger = logging.getLogger('training_logger')
        self.logger.setLevel(logging.INFO)
        self.setup_resfiles()

        log_file = f"./res/log/{log_file}.log"
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # 根据任务类型定义不同的字符串模版
        templates = {
            'multiclass': ('Round = {runId}, BestEpoch = {best_epoch:02}, '
                    'Acc = {Acc:.4f}, F1 = {F1:.4f}, P = {P:.4f}, Recall = {Recall:.4f}, AUROC = {AUROC:.4f}, '
                    'Time = {sum_time:.2f} s'),
            'rec': ('Round = {runId}, BestEpoch = {best_epoch:02}, '
                    'MAE = {MAE:.4f}, RMSE = {RMSE:.4f}, NMAE = {NMAE:.4f}, NRMSE = {NRMSE:.4f}, '
                    'Acc@1 = {Acc1:.4f}, Acc@5 = {Acc5:.4f}, Acc@10 = {Acc10:.4f}, '
                    'Time = {sum_time:.2f}s')
        }

        if task_type not in templates:
            raise ValueError("Unsupported task type specified.")
        
        self.msg_template = templates[task_type]
    
    def setup_resfiles(self):
        res_base_dir = './res'
        subdirs = ['csv', 'fig', 'log', 'weight']
        for subdir in subdirs:
            dir_path = os.path.join(res_base_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
    
    def base_msg(self, runId, tracker, results, sum_time):
        results = { 
            **results,
            'runId': runId,
            'best_epoch': tracker.best_epoch + 1,
            'sum_time': sum_time
        }
        msg = self.msg_template.format(**results)
        return msg
    
    def print_results(self, runId, tracker, results, sum_time):
        msg = self.base_msg(runId, tracker, results, sum_time)    
        res_msg = f"Test Result: {msg}"
        self.logger.info(res_msg)

    def print_epoch_results(self, runId, tracker, results, sum_time, epoch, epochs, epoch_loss):
        msg = self.base_msg(runId, tracker, results, sum_time)
        epoch_msg = f"epoch=[{epoch + 1}/{epochs}], {msg}, vLoss={epoch_loss:.4f}"
        self.logger.info(epoch_msg)

    def info(self, msg):
        self.logger.info(msg)