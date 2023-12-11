from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from DCNN.loss import si_snr
from DCNN.utils.eval_utils import ild_db, ipd_rad, speechMask
from DCNN.feature_extractors import Stft, IStft
from DCNN.datasets.base_dataset import BaseDataset
import torch.nn as nn
import torch
from mbstoi import mbstoi
import warnings
warnings.simplefilter('ignore')
import torch
from torchmetrics import SignalNoiseRatio

snr = SignalNoiseRatio()


class EvalMetrics(nn.Module):
    def __init__(self, win_len=400, win_inc=100, fft_len=512) -> None:
        super().__init__()

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.fbins = int(fft_len/2 + 1)
        self.stft = Stft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
        self.istft = IStft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
        self.stoi = ShortTimeObjectiveIntelligibility(fs=16000)

    def forward(self,NOISY_DATASET_PATH, CLEAN_DATASET_PATH,ENHANCED_DATASET_PATH, testset_len=5,SR=16000):

        dataset = BaseDataset(NOISY_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)
        datasetEn = BaseDataset(ENHANCED_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)
        breakpoint()

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False)
        
        dataloaderEn = torch.utils.data.DataLoader(
        datasetEn,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

        dataloader = iter(dataloader)
        dataloaderEn = iter(dataloaderEn)
        # testset_len = len(dataloader)
        

        noisy_snr_l = torch.zeros((testset_len))


        noisy_snr_r = torch.zeros((testset_len))

        enhanced_snr_l = torch.zeros((testset_len))
        enhanced_snr_r = torch.zeros((testset_len))

        noisy_stoi_l = torch.zeros((testset_len))
        noisy_stoi_r = torch.zeros((testset_len))

        enhanced_stoi_l = torch.zeros((testset_len))
        enhanced_stoi_r = torch.zeros((testset_len))

        masked_ild_error = torch.zeros((testset_len, self.fbins))
        masked_ipd_error = torch.zeros((testset_len, self.fbins))

        noisy_mbstoi = torch.zeros((testset_len))
        enhanced_mbstoi = torch.zeros((testset_len))

        avg_snr = torch.zeros(testset_len)
        
        for i in range(testset_len):  # Enhance 10 samples
            try:
                batch = next(dataloader)
                batchEn = next(dataloaderEn)
            except StopIteration:
                break


            noisy_samples = (batch[0])
            clean_samples = (batch[1])[0]
            # model_output = model(noisy_samples)[0]
            model_output = (batchEn[0])[0]
            clean_samples=(clean_samples)/(torch.max(clean_samples))
            # model_output=(model_output)/(torch.max(model_output))
            # mo = model_output.numpy()
        

            # breakpoint()
            
            
            # noisy_snr_l[i] = si_snr(noisy_samples[0][0, :], clean_samples[0, :])
            # noisy_snr_r[i] = si_snr(noisy_samples[0][1, :], clean_samples[1, :])

            # enhanced_snr_l[i] = si_snr(model_output[0, :], clean_samples[0, :])
            # enhanced_snr_r[i] = si_snr(model_output[1, :], clean_samples[1, :])
            
            # noisy_snr_l[i] = snr(noisy_samples[0][0, :], clean_samples[0, :])
            # noisy_snr_r[i] = snr(noisy_samples[0][1, :], clean_samples[1, :])

            # enhanced_snr_l[i] = snr(model_output[0, :], clean_samples[0, :])
            # enhanced_snr_r[i] = snr(model_output[1, :], clean_samples[1, :])
            
            # noisy_stoi_l[i] = self.stoi(noisy_samples[0][0, :], clean_samples[0, :])
            # noisy_stoi_r[i] = self.stoi(noisy_samples[0][1, :], clean_samples[1, :])

            # enhanced_stoi_l[i] = self.stoi(model_output[0, :], clean_samples[0, :])
            # enhanced_stoi_r[i] = self.stoi(model_output[1, :], clean_samples[1, :])

            noisy_stft_l = self.stft(noisy_samples[0][0, :])
            noisy_stft_r = self.stft(noisy_samples[0][1, :])

            enhanced_stft_l = self.stft(model_output[0, :])
            enhanced_stft_r = self.stft(model_output[1, :])

            target_stft_l = self.stft(clean_samples[0, :])
            target_stft_r = self.stft(clean_samples[1, :])
            

            

            mask = speechMask(target_stft_l,target_stft_r).squeeze(0)
            target_ild = ild_db(target_stft_l.abs(), target_stft_r.abs())
            enhanced_ild = ild_db(enhanced_stft_l.abs(), enhanced_stft_r.abs())

            target_ipd = ipd_rad(target_stft_l, target_stft_r)
            enhanced_ipd = ipd_rad(enhanced_stft_l, enhanced_stft_r)

            ild_loss = (target_ild - enhanced_ild).abs()

            ipd_loss = (target_ipd - enhanced_ipd).abs()
            
            mask_sum=mask.sum(dim=1)
            mask_sum[mask_sum==0]=1
        
            masked_ild_error[i,:] = (ild_loss*mask).sum(dim=1)/ mask_sum
            masked_ipd_error[i,:] = (ipd_loss*mask).sum(dim=1)/ mask_sum
            
            avg_snr[i] = (noisy_snr_l[i] + noisy_snr_r[i])/2
        
            # noisy_signals = noisy_samples[0].detach().cpu().numpy()
            # clean_signals = clean_samples.detach().cpu().numpy()
            # enhanced_signals = model_output.detach().cpu().numpy()
            # noisy_mbstoi[i] = mbstoi(clean_signals[0,:], clean_signals[1,:],
            #         noisy_signals[0,:], noisy_signals[1,:], fsi=SR)
            # enhanced_mbstoi[i] = mbstoi(clean_signals[0,:], clean_signals[1,:],
            #         enhanced_signals[0,:], enhanced_signals[1,:], fsi=SR)
            
            print('Processed Signal ', i+1 , ' of ', testset_len)

    

        
        # improved_snr_l = (enhanced_snr_l - noisy_snr_l)
        # improved_snr_r = (enhanced_snr_r - noisy_snr_r)

        # improved_stoi_l = (enhanced_stoi_l - noisy_stoi_l)
        # improved_stoi_r = (enhanced_stoi_r - noisy_stoi_r)

        # improved_mbstoi = enhanced_mbstoi - noisy_mbstoi

        return masked_ild_error, masked_ipd_error
