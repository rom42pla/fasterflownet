from models.fastflownet import FastFlowNet
from models.profile import profile

f_pwc, p_pwc = profile(FastFlowNet(), input_size=(1, 6, 384, 768), device='cuda')

# print('PWCNet: \tflops(G)/params(M):%.1f/%.2f'%(f_pwc/1e9,p_pwc/1e6))
print(f_pwc / 1e9)
