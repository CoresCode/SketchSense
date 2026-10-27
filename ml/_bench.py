import time, torch, torch.nn as nn
from model import build, count_params
dev = torch.device("cuda")
m = build(83).to(dev).to(memory_format=torch.channels_last)
print("params", round(count_params(m)/1e6, 2), "M")
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda")
for bs in (128, 256, 512):
    x = torch.randn(bs,1,64,64, device=dev).to(memory_format=torch.channels_last)
    y = torch.randint(0,83,(bs,), device=dev)
    for _ in range(3):  # warmup
        with torch.amp.autocast("cuda"):
            l = lossf(m(x), y)
        scaler.scale(l).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(10):
        with torch.amp.autocast("cuda"):
            l = lossf(m(x), y)
        scaler.scale(l).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time()-t)/10
    print(f"bs={bs:<4} {dt*1000:6.1f} ms/step  {bs/dt:7.0f} img/s  -> epoch of 1M in {1_000_000/(bs/dt)/60:.1f} min", flush=True)
print(torch.cuda.get_device_properties(0).name, "| mem", round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB")
