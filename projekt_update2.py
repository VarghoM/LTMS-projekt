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

__kernel void R(
__global float *clDataInR, 
__global float *clDataInPr, 
float r_s,
float dt,
int N)
{
    int gid = get_global_id(0);

    if (gid >= N) {
        return;
    }

    float rad = clDataInR[gid];
    float p_r = clDataInPr[gid];

    float a = rad/(rad - r_s);

    clDataInR[gid] += dt * ( p_r / a );
}

// Tavaline Euleri meetod
__kernel void Euler(
__global float *clDataInTht, 
__global float *clDataInPtht, 
__global float *clDataInR,
__global float *clDataInPr, 
__global float *clDataInFii,
__global float *clDataInPfii, 
__global float *clDataInT,
float r_s,
float L, 
float dt,
int N)
{
    int gid = get_global_id(0);

    if (gid >= N) {
        return;
    }

    float theta = clDataInTht[gid];
    float p_tht = clDataInPtht[gid];

    float r = clDataInR[gid];
    float p_r = clDataInPr[gid];

    float fii = clDataInFii[gid];
    float p_fii = clDataInPfii[gid];

    float t = clDataInT[gid];

    // 1 / (1 - (r_s / r)) = r/(r - r_s)
    float a = r/(r - r_s);
    float da_dr = - r_s / ( (r - r_s)*(r - r_s) );

    clDataInTht[gid] += dt * (p_tht/(r*r));
    clDataInPtht[gid] += dt * ( (p_fii * p_fii) / (r * r) ) * ( cos(theta) / ( sin(theta) * sin(theta) * sin(theta) ) );

    clDataInR[gid] += dt * ( p_r / a );
    clDataInPr[gid] += dt * 0.5 * ( (da_dr * p_r * p_r) / (a * a) + (2.0 * p_fii * p_fii)/(r * r * r) );

    clDataInFii[gid] += dt * ( p_fii / (r * r) );

    clDataInT[gid] += dt * ( (p_r*p_r) / a   +  (p_fii * p_fii) / (r * r) );

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
def sim_Euler(arr, Blocksize, Outstep, Step, Number, ctx, queue, prg, max_steps, draw = True):
       
    time_start = time.time()
    print('Euler')

    tht_host = arr[0][0]
    ptht_host = arr[0][1]
    r_host = arr[1][0]
    pr_host = arr[1][1]    
    fii_host = arr[2][0]
    pfii_host = arr[2][1]
    t_host = arr[3]
    r_s = arr[4]

    knl_euler = prg.Euler

    # Reserveerin mälu, määran kasutuse
    mf = cl.mem_flags
    tht_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = tht_host)
    ptht_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = ptht_host)
    r_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = r_host)
    pr_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = pr_host)
    fii_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = fii_host)
    pfii_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = pfii_host)
    t_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = t_host)

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
        m = "fii"
        if m == "tht":
            i = 0
            folder = "kaadrid_Euler" + str(int(time.time()))
            os.makedirs(folder, exist_ok = True)

            fig, ax = plt.subplots(figsize=(10, 10))
            sc = ax.scatter(np.zeros(Number), np.zeros(Number), c = r_host, cmap = "jet", s=20, alpha=0.8) # viridis asemel turbo
            sc.set_clim(vmin=0, vmax=100)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Osakese raadius $r$')
            ax.set_xlabel(r'$\theta$')
            ax.set_ylabel(r'$p_\theta$')
            ax.set_xlim(-2*np.pi, 2*np.pi)
            ax.set_ylim(-10, 10)
            info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
        elif m == "fii":
            i = 0
            folder = "kaadrid_Euler" + str(int(time.time()))
            os.makedirs(folder, exist_ok = True)

            fig, ax = plt.subplots(figsize=(10, 10))
            sc = ax.scatter(np.zeros(Number), np.zeros(Number), c = tht_host, cmap = "jet", s=20, alpha=0.8) # viridis asemel turbo
            sc.set_clim(vmin=0, vmax=5)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(r'$\theta$')
            ax.set_xlabel(r'$\fii')
            ax.set_ylabel('$r$')
            ax.set_xlim(-2*np.pi, 2*np.pi)
            ax.set_ylim(0, 100)
            info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))  

    try:
        while not finish:

            if draw == True and n % Outstep == 0:
                if m == "tht":
                    cl.enqueue_copy(queue, tht_host, tht_dev)
                    cl.enqueue_copy(queue, ptht_host, ptht_dev)
                    cl.enqueue_copy(queue, r_host, r_dev)

                    sc.set_offsets(np.c_[tht_host.copy(), ptht_host.copy()])
                    sc.set_array(r_host.copy())

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
                
                elif m == "fii":
                    cl.enqueue_copy(queue, tht_host, tht_dev)
                    cl.enqueue_copy(queue, r_host, r_dev)
                    cl.enqueue_copy(queue, fii_host, fii_dev)

                    sc.set_offsets(np.c_[fii_host.copy(), r_host.copy()])
                    sc.set_array(tht_host.copy())

                    textbox = (
                        f't = {Step * n:.2f} s\n'
                        f'N = {Number}\n'
                        f'r = [{tht_host.min():.1f} ... {tht_host.max():.1f}]\n'
                        fr'$p_\phi$ = [{pfii_host.min():.1f} ... {pfii_host.max():.1f}]'
                    )
                    info_text.set_text(textbox)

                    f = f"graph_{i:04d}.png"
                    path = os.path.join(folder, f)
                    plt.savefig(path, dpi = 150, bbox_inches='tight')

                    i += 1

            # arvutused kernelis
            knl_euler(queue, gws, lws, tht_dev, ptht_dev, r_dev, pr_dev, fii_dev, pfii_dev, t_dev, np.int32(r_s), np.int32(1), Step, np.int32(Number))

            n += 1

            if n >= max_steps:
                finish = True

    # Ctrl + C
    except KeyboardInterrupt:
        print("Interrupted")

    cl.enqueue_copy(queue, tht_host, tht_dev)
    cl.enqueue_copy(queue, ptht_host, ptht_dev)
    cl.enqueue_copy(queue, r_host, r_dev)
    cl.enqueue_copy(queue, pr_host, pr_dev)
    cl.enqueue_copy(queue, fii_host, fii_dev)
    cl.enqueue_copy(queue, t_host, t_dev)

    # prindib tht ja ptht normid
    print(np.linalg.norm(tht_host))
    print(np.linalg.norm(ptht_host))

    tht_arr = tht_host.copy()
    ptht_arr = ptht_host.copy()

    tht_dev.release()
    ptht_dev.release()
    r_dev.release()
    pr_dev.release()
    fii_dev.release()
    t_dev.release()
   
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
    r_s = np.int32(1)
    pr_host = np.linspace(-0.2, -0.1, Number, dtype=np.float32)
    print('mod Euler')

    knl_tht = prg.Theta
    knl_ptht = prg.Ptht
    knl_r = prg.R
 
    # Reserveerin mälu, määran kasutuse
    mf = cl.mem_flags
    tht_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = tht_host)
    ptht_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = ptht_host)
    r_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = r_host)
    pfii_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = pfii_host)
    pr_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = pr_host)


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

            knl_tht(queue, gws, lws, tht_dev, ptht_dev, r_dev, Step, np.int32(Number))
            knl_ptht(queue, gws, lws, tht_dev, ptht_dev, r_dev, pfii_dev, Step, np.int32(Number))
            knl_r(queue, gws, lws, r_dev, pr_dev, r_s, Step, np.int32(Number))

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
    r_dev.release()
    pr_dev.release()

   

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
    Number = 5
    # Nurga algväärtus
    Tht = np.pi/2+0.5
    # Impulsi algväärtus
    Ptht = 0
    # Samm
    Step = np.float32(1/32)
    # Blocksize
    Blocksize = 512
    # Outstep
    Outstep = 100

    # Sammude arv kuni süsteem "stabiliseerub" 50 000, siis jõuavad kõige kaugemad osakesed tagasi algusesse ka
    max_steps = 5000

    # Algväärtused

    # raadius
    r_host = np.linspace(5, 50, Number, dtype=np.float32)
    # nullid või väike vahemik 0 ümber
    #pr_host = np.zeros(Number, dtype=np.float32)
    pr_host = np.flip(np.linspace(0.2, 0.4, Number, dtype=np.float32))
    pr_host = pr_host.copy()

    # polaarnurk
    tht_host = np.full(Number, Tht, dtype=np.float32)
    #ptht_host = np.full(Number, Ptht, dtype=np.float32)
    #tht_host = np.linspace(Tht-0.5, Tht+0.5, Number, dtype=np.float32)
    ptht_host = np.linspace(-5, 5, Number, dtype=np.float32)
    ptht_host = ((np.random.rand(Number)*2 - 1) / np.sqrt(r_host)).astype(np.float32)

    # asimuudi nurk
    fii_host = np.linspace(-0.1, 0.1, Number, dtype=np.float32)
    pfii_host = np.linspace(0.3, 0.8, Number, dtype=np.float32)

    # aeg
    t_host = np.zeros(Number, dtype=np.float32)
    # pt ei ole vajalik teiste jaoks

    # Schwarzschildi raadius
    r_s = 1

    hosts = [[tht_host, ptht_host], [r_host, pr_host], [fii_host, pfii_host], t_host, r_s]

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
    Euler = sim_Euler(hosts, Blocksize, Outstep, Step, Number, ctx, queue, prg, max_steps)
    #Euler_mod = sim_Euler_mod(tht_host, ptht_host, r_host, pfii_host, Blocksize, Outstep, Step, Number, ctx, queue, prg, max_steps)

    print('Euler runtime ' + str(Euler["runtime"]))
    #print('Euler mod runtime ' + str(Euler_mod["runtime"]))



    print('end')


