# eeskuju:
# https://github.com/inducer/pyopencl/blob/main/examples/n-body.py
# http://manuelhohmann.ddns.net/ut/teaching/gpu.pdf 

# loob video:
# ffmpeg -framerate 10 -i graph_%04d.png seeria.mp4

import time
import os

import numpy as np

import pyopencl as cl

import matplotlib.pyplot as plt


#platforms = cl.get_platforms()
#print(platforms)
# mul on [<pyopencl.Platform 'NVIDIA CUDA' at 0x20326cc8cd0>, <pyopencl.Platform 'Intel(R) OpenCL HD Graphics' at 0x203248b0420>]



# Siia tuleb OpenCL kood:
BlobOpenCL = """

// Modifitseeritud Euleri meetodi kernelid, loodud võrrandite (13), (14) põhjal (https://export.arxiv.org/pdf/2201.04694)
__kernel void Theta(
__global float *clDataInTht, 
__global float *clDataInPtht, 
__global float *r, 
float dt,
int N)
{
    // Vaadeldava osakese indeks
    int gid = get_global_id(0);

    // Kui indeks on suurem kui osakeste arv ise siis ignoreerib
    if (gid >= N) {
        return;
    }

    float p_tht = clDataInPtht[gid];
    float radius = r[gid];


    clDataInTht[gid] += dt * (p_tht/(radius*radius));
}

__kernel void Ptht(
__global float *clDataInTht, 
__global float *clDataInPtht, 
__global float *r, 
__global float *Pfii , 
float dt,
int N)
{
    int gid = get_global_id(0);

    if (gid >= N) {
        return;
    }

    float p_fii = Pfii[gid];
    float radius = r[gid];
    float theta = clDataInTht[gid];

    clDataInPtht[gid] += dt * ( (p_fii * p_fii) / (radius * radius) ) * ( cos(theta) / ( sin(theta) * sin(theta) * sin(theta) ) );
}

// Tavaline Euleri meetod
__kernel void Euler(
__global float *clDataInTht, 
__global float *clDataInPtht, 
__global float *r, 
__global float *Pfii, 
float dt,
int N)
{
    int gid = get_global_id(0);

    if (gid >= N) {
        return;
    }

    float p_fii = Pfii[gid];
    float radius = r[gid];
    float theta = clDataInTht[gid];
    float p_tht = clDataInPtht[gid];

    clDataInTht[gid] += dt * (p_tht/(radius*radius));
    clDataInPtht[gid] += dt * ( (p_fii * p_fii) / (radius * radius) ) * ( cos(theta) / ( sin(theta) * sin(theta) * sin(theta) ) );
}

"""

# Valib platvormi
def select_platform(name_part):
    platforms = cl.get_platforms()
    for p in platforms:
        if name_part.lower() in p.name.lower():
            return p
    return platforms[0] # Kui ei leia, võta esimene

# Panna draw = True, et graafikuid teeks
def select_platform(name_part):
    platforms = cl.get_platforms()
    for p in platforms:
        if name_part.lower() in p.name.lower():
            return p
    return platforms[0] # Kui ei leia, võta esimene

# Panna draw = True, et graafikuid teeks
def sim_Euler(tht_host, ptht_host, r_host, pfii_host, Blocksize, Outstep, Step, Number, ctx, queue, prg, max_steps, draw = True):
    time_start = time.time()
    print('Euler')

    knl_euler = prg.Euler

    # Reserveerin mälu, määran kasutuse
    mf = cl.mem_flags
    tht_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = tht_host)
    ptht_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = ptht_host)
    r_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = r_host)
    pfii_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = pfii_host)

    # Local workgroup size
    lws = (Blocksize,)
    # Global workgoup size
    gws = (int(np.ceil(Number / Blocksize) * Blocksize),)
    
    finish = False
    n = 0

    tht_arr = []
    ptht_arr = []

    # loob keskkonna graafikute joonistamiseks
    if draw == True:
        i = 0
        folder = "kaadrid_Euler" + str(int(time.time()))
        os.makedirs(folder, exist_ok = True)

        fig, ax = plt.subplots(figsize=(10, 10))
        sc = ax.scatter(np.zeros(Number), np.zeros(Number), c = r_host, cmap = "viridis", s=20, alpha=0.8) # viridis asemel turbo
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Osakese raadius $r$')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$p_\theta$')
        ax.set_xlim(-1, np.pi)
        ax.set_ylim(-10, 10)
        info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    try:
        while not finish:

            if draw == True and n % Outstep == 0:
                cl.enqueue_copy(queue, tht_host, tht_dev)
                cl.enqueue_copy(queue, ptht_host, ptht_dev)

                sc.set_offsets(np.c_[tht_host.copy(), ptht_host.copy()])

                textbox = (
                    f't = {Step * n:.2f} s\n'
                    f'N = {Number}\n'
                    f'r = [{r_host.min():.1f} ... {r_host.max():.1f}]\n'
                    fr'$p_\phi$ = [{pfii_host.min():.1f} ... {pfii_host.max():.1f}]'
                )
                info_text.set_text(textbox)

                f = f"graph_{i:04d}.png"
                path = os.path.join(folder, f)
                plt.savefig(path, dpi = 150, bbox_inches='tight')

                i += 1

            # arvutused kernelis
            knl_euler(queue, gws, lws, tht_dev, ptht_dev, r_dev, pfii_dev, Step, np.int32(Number))

            n += 1

            if n >= max_steps:
                finish = True

    # Ctrl + C
    except KeyboardInterrupt:
        print("Interrupted")

    cl.enqueue_copy(queue, tht_host, tht_dev)
    cl.enqueue_copy(queue, ptht_host, ptht_dev)

    # prindib tht ja ptht normid
    print(np.linalg.norm(tht_host))
    print(np.linalg.norm(ptht_host))

    tht_arr = tht_host.copy()
    ptht_arr = ptht_host.copy()

    tht_dev.release()
    ptht_dev.release()
   
    if draw == True:
        plt.close(fig)

    return {
        "runtime": time.time() - time_start,
        "tht": tht_arr,
        "ptht": ptht_arr,
        "t": [n, Step]
    }




def sim_Euler_mod(tht_host, ptht_host, r_host, pfii_host, Blocksize, Outstep, Step, Number, ctx, queue, prg, max_steps):
    time_start = time.time()

    print('mod Euler')

    knl_tht = prg.Theta
    knl_ptht = prg.Ptht
 
    # Reserveerin mälu, määran kasutuse
    mf = cl.mem_flags
    tht_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = tht_host)
    ptht_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = ptht_host)
    r_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = r_host)
    pfii_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = pfii_host)

    # Local workgroup size
    lws = (Blocksize,)
    # Global workgoup size
    gws = (int(np.ceil(Number / Blocksize) * Blocksize),)
    
    finish = False
    n = 0

    tht_arr = []
    ptht_arr = []

    try:
        while not finish:
            if n % Outstep == 0:
                cl.enqueue_copy(queue, tht_host, tht_dev)
                cl.enqueue_copy(queue, ptht_host, ptht_dev)


            knl_tht(queue, gws, lws, tht_dev, ptht_dev, r_dev, Step, np.int32(Number))
            knl_ptht(queue, gws, lws, tht_dev, ptht_dev, r_dev, pfii_dev, Step, np.int32(Number))

            n += 1

            if n >= max_steps:
                finish = True

    except KeyboardInterrupt:
        print("Interrupted")

    # prindib tht ja ptht normid
    print(np.linalg.norm(tht_host))
    print(np.linalg.norm(ptht_host))

    tht_dev.release()
    ptht_dev.release()

    tht_arr = tht_host
    ptht_arr = ptht_host   

    return {
        "runtime": time.time() - time_start,
        "tht": tht_arr,
        "ptht": ptht_arr,
        "t": [n, Step]
    }


if __name__ == "__main__":
  

    # Osakeste arv
    Number = 50
    # Nurga algväärtus
    Tht = np.pi/2+1
    # Impulsi algväärtus
    Ptht = 0+5
    # Samm
    Step = np.float32(1/32)
    # Blocksize
    Blocksize = 512
    # Outstep
    Outstep = 50

    # Sammude arv kuni süsteem "stabiliseerub" 50 000, siis jõuavad kõige kaugemad osakesed tagasi algusesse ka
    max_steps = 5000

    # Algväärtused
    tht_host = np.full(Number, Tht, dtype=np.float32)
    ptht_host = np.full(Number, Ptht, dtype=np.float32)

    r_host = np.linspace(10, 50, Number).astype(np.float32)
    pfii_host = np.linspace(1, 5, Number).astype(np.float32)

    

    # Valin NVIDIA
    platform = select_platform("NVIDIA")
    # Valin GPU
    devices = platform.get_devices(device_type=cl.device_type.GPU)

    # Loon konteksti
    ctx = cl.Context(devices)
    queue = cl.CommandQueue(ctx)

    print("Device : %s" % devices)
    print("Number of particles : %s" % Number)
    print("Step of iteration : %s" % Step)

    # Kompileerimine
    prg = cl.Program(ctx, BlobOpenCL).build()

    # kaks meetodit, #-id eest võtta et võrrelda
    Euler = sim_Euler(tht_host, ptht_host, r_host, pfii_host, Blocksize, Outstep, Step, Number, ctx, queue, prg, max_steps)
    #Euler_mod = sim_Euler_mod(tht_host, ptht_host, r_host, pfii_host, Blocksize, Outstep, Step, Number, ctx, queue, prg, max_steps)

    print('Euler runtime ' + str(Euler["runtime"]))
    #print('Euler mod runtime ' + str(Euler_mod["runtime"]))



    print('end')


