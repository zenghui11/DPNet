import torch
import torch.nn.functional as F

def fft_transform(image):
    fft_result = torch.fft.fft2(image)
    # fft_shifted = torch.fft.fftshift(fft_result)
    return fft_result 

def get_magnitude_phase(fft_image):
    magnitude = torch.abs(fft_image)
    phase = torch.angle(fft_image)
    return magnitude, phase


def magnitude_loss(fused_mag, infrared_mag, visible_mag, alpha=0.5): #实部结构 红外调大一点
    target_mag = alpha * infrared_mag + (1 - alpha) * visible_mag
    loss = F.l1_loss(fused_mag, target_mag)

    return loss

def phase_loss(fused_phase, infrared_phase, visible_phase, alpha=0.5):
    target_phase = alpha * infrared_phase + (1 - alpha) * visible_phase
    loss = F.l1_loss(fused_phase, target_phase)
    return loss

def total_loss(fused_image, infrared_image, visible_image, alpha=0.5, beta=0.5, gamma=0.5):
    fused_fft = fft_transform(fused_image)
    infrared_fft = fft_transform(infrared_image)
    visible_fft = fft_transform(visible_image)
    
    fused_mag, fused_phase = get_magnitude_phase(fused_fft)
    infrared_mag, infrared_phase = get_magnitude_phase(infrared_fft)
    visible_mag, visible_phase = get_magnitude_phase(visible_fft)
    
    # weight_matrix = create_weight_matrix(fused_mag.shape[-2:], low_freq_weight=1.0, high_freq_weight=0.5)
    # weight_matrix = weight_matrix.to(fused_image.device)
    
    mag_loss = magnitude_loss(fused_mag, infrared_mag, visible_mag, alpha)
    pha_loss = phase_loss(fused_phase, infrared_phase, visible_phase, alpha)
    
    total = beta * mag_loss + gamma * pha_loss
    return total
