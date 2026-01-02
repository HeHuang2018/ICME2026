import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np
import random
from scipy.special import comb
import torch
import torch.fft
import cv2
import matplotlib.pyplot as plt
vrange=(0.,1.)
def phase_attention_with_Bilateral(amplitude, phase):
    constant_amplitude = torch.ones_like(amplitude)
    reconstructed_from_phase = torch.fft.ifftn(torch.polar(constant_amplitude, phase)).real
    reconstructed_from_phase = torch.sqrt(reconstructed_from_phase ** 2).detach().cpu().numpy()
    bilateral = cv2.bilateralFilter(src=reconstructed_from_phase, d=3, sigmaColor=75, sigmaSpace=75)
    phase_attention = (bilateral - np.min(bilateral)) / (np.max(bilateral) - np.min(bilateral))
    return np.expand_dims(phase_attention, axis=-1)

def sector_mask(amplitude, center, radius, angle_range=(0, 360)):
    y, x = np.ogrid[:amplitude.shape[0], :amplitude.shape[1]]

    # Calculate the distance from the center
    distance_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Calculate the angle (in degrees) for each point with respect to center
    angles = np.rad2deg(np.arctan2(y - center[1], x - center[0])) % 360

    # Create mask where distance is less than the radius and within the angle range
    mask = (distance_from_center <= radius) & (angles >= angle_range[0]) & (angles <= angle_range[1])

    return mask


def amplitude_masking(amplitude, phase):
    re_amp = amplitude.clone()

    angles = np.linspace(0, 2*np.pi, 100)
    h_theta = np.zeros_like(angles)

    for i, theta in enumerate(angles): # radian을 도로 변환  => rad * 180 / np.pi = 도
        sum_freq = 0
        for r in range(1, amplitude.shape[0]):
            x = int(r * np.cos(theta))
            y = int(r * np.sin(theta))
            sum_freq += amplitude[x, y]
        h_theta[i] = sum_freq

    h_theta = np.abs(h_theta)  # h_theta /= np.sum(h_theta)

    max_angle = angles[np.where(h_theta == h_theta.max())][0] * 180 / np.pi
    min_angle = angles[np.where(h_theta == h_theta.min())][0] * 180 / np.pi

    degree = np.random.randint(60)  # 최대 각도 60도로 설정
    center = (amplitude.shape[0] // 2, amplitude.shape[1] // 2)
    max_angle_start, max_angle_end = max_angle - (degree/2), max_angle + (degree/2)
    min_angle_start, min_angle_end = min_angle - (degree/2), min_angle + (degree/2)

    if max_angle_end > 360:
        max_angle_end = (max_angle_end - 360)
    max_angle_range = (max_angle_start, max_angle_end)

    if min_angle_end > 360:
        min_angle_end = (min_angle_end - 360)
    min_angle_range = (min_angle_start, min_angle_end)

    radius = np.random.randint(amplitude.shape[0] // 4)  # 최대 반지름 48로 설정

    ###### Min & Max masking code ######
    max_mask = 1 - sector_mask(amplitude, center, radius, max_angle_range)
    min_mask = 1 - sector_mask(amplitude, center, radius, min_angle_range)

    masked_amp = amplitude * torch.from_numpy(np.expand_dims(max_mask, axis=-1))
    masked_amp = torch.fft.ifftshift(masked_amp)
    max_mask_reconstruction = torch.fft.ifftn(torch.polar(masked_amp, phase)).real

    ##### Min & Max switching code #####
    max_coord, min_coord = np.where(max_mask == False), np.where(min_mask == False)
    if max_coord[0].shape < min_coord[0].shape:
        x_min_coord, y_min_coord = min_coord[0][:len(max_coord[0])], min_coord[1][:len(max_coord[1])]
        re_amp[x_min_coord, y_min_coord] = amplitude[max_coord]
        re_amp[max_coord] = amplitude[x_min_coord, y_min_coord]
    else:
        x_max_coord, y_max_coord = max_coord[0][:len(min_coord[0])], max_coord[1][:len(min_coord[1])]
        re_amp[x_max_coord, y_max_coord] = amplitude[min_coord]
        re_amp[min_coord] = amplitude[x_max_coord, y_max_coord]
    re_amp = torch.fft.ifftshift(re_amp)
    max_min_switching_reconstruction = torch.fft.ifftn(torch.polar(re_amp, phase)).real

    return max_mask_reconstruction, max_min_switching_reconstruction
    
def FourierAugmentativeTransformer(image):  # image=[192, 192, 1]
    fft_res = torch.fft.fftn(image, dim=(0, 1))
    amplitude = torch.abs(torch.fft.fftshift(fft_res))  # 0 주파수 성분을 이미지의 중앙으로 이동
    phase = torch.angle(fft_res)

    if 0.5 > np.random.random():  # Reversing the histogram distribution of the amplitude
        amplitude = np.median(amplitude) - amplitude
    amp_masking, amp_intra_modulation = amplitude_masking(amplitude=amplitude, phase=phase)
    phase_attention = phase_attention_with_Bilateral(amplitude, phase)

    aug_img = (0.5 * amp_masking) + (0.5 * amp_intra_modulation)
    aug_img = aug_img * phase_attention + aug_img
    aug_img = aug_img.detach().cpu().numpy()

    scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
    location = np.array(random.gauss(0, 0.5), dtype=np.float32)
    aug_img = np.clip(aug_img * scale + location, vrange[0], vrange[1])
    aug_img = (image + aug_img) / 2
    return aug_img



import torch
import numpy as np

from einops import rearrange
from opt_einsum import contract


class GeneralFourierOnline(torch.nn.Module):

    def __init__(
            self, img_size, groups, phases, f_cut=1, phase_cut=1, min_str=0, mean_str=5, granularity=64
    ):
        super().__init__()

        _x = torch.linspace(- img_size / 2, img_size / 2, steps=img_size)
        self._x, self._y = torch.meshgrid(_x, _x, indexing='ij')

        self.groups = groups
        self.num_groups = len(groups)
        self.freqs = [f / img_size for f in groups]

        self.phase_range = phases
        self.num_phases = granularity
        self.phases = - np.pi * np.linspace(phases[0], phases[1], num=granularity)

        self.f_cut = f_cut
        self.phase_cut = phase_cut

        self.min_str = min_str
        self.mean_str = mean_str

        self.eps_scale = img_size / 32

    def sample_f_p(self, b, c, device):
        f_cut = self.f_cut
        p_cut = self.phase_cut

        freqs = torch.tensor(self.freqs, device=device, dtype=torch.float32)
        phases = torch.tensor(self.phases, device=device, dtype=torch.float32)

        f_s = freqs[
            torch.randint(0, self.num_groups, (b, c, f_cut, 1), device=device)
        ]

        p_s = phases[
            torch.randint(0, self.num_phases, (b, c, f_cut, p_cut), device=device)
        ]

        return f_s, p_s, f_cut, p_cut

    def forward(self, x):
        init_shape = x.shape
        if len(x.shape) < 4:
            x = rearrange(x, 'c h w -> () c h w')
        b, c, h, w = x.shape

        freqs, phases, num_f, num_p = self.sample_f_p(b, c, x.device)
        strengths = torch.empty_like(phases).exponential_(1 / self.mean_str) + self.min_str

        return self.apply_fourier_aug(freqs, phases, strengths, x).reshape(init_shape)

    def apply_fourier_aug(self, freqs, phases, strengths, x):
        aug = contract(
            'b c f p, b c f p h w -> b c h w',
            strengths,
            self.gen_planar_waves(freqs, phases, x.device)
        )
        aug *= 1 / (self.f_cut * self.phase_cut)
        return torch.clamp(x + aug, 0, 1)

    def gen_planar_waves(self, freqs, phases, device):
        _x, _y = self._x.to(device), self._y.to(device)
        freqs, phases = rearrange(freqs, 'b c f p -> b c f p () ()'), rearrange(phases, 'b c f p -> b c f p () ()')
        _waves = torch.sin(
            2 * torch.pi * freqs * (
                    _x * torch.cos(phases) + _y * torch.sin(phases)
            ) - torch.pi / 4
        )
        _waves.div_(_waves.norm(dim=(-2, -1), keepdim=True))

        return self.eps_scale * _waves

    def __str__(self):
        return f'GeneralFourierOnline(' \
               f'f={self.groups}, phases={self.phase_range}, ' \
               f'f_cut={self.f_cut}, p_cut={self.phase_cut}' \
               f', min_str={self.min_str}, max_str={self.mean_str}' \
               f')'