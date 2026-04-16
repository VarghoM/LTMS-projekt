# eeskuju:
# https://github.com/inducer/pyopencl/blob/main/examples/n-body.py
# http://manuelhohmann.ddns.net/ut/teaching/gpu.pdf 

import getopt
import sys
import time

import numpy as np

import pyopencl as cl
import pyopencl.tools 
import pyopencl.array


#platforms = cl.get_platforms()
#print(platforms)
# mul on [<pyopencl.Platform 'NVIDIA CUDA' at 0x20326cc8cd0>, <pyopencl.Platform 'Intel(R) OpenCL HD Graphics' at 0x203248b0420>]



# Siia tuleb OpenCL kood:
BlobOpenCL = """

// Euleri meetodi kernelid
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

"""

# Valib platvormi
def select_platform(name_part):
    platforms = cl.get_platforms()
    for p in platforms:
        if name_part.lower() in p.name.lower():
            return p
    return platforms[0] # Kui ei leia, võta esimene

# Kontrollimata / pooleli
#def save(tht, ptht, r, pfii, dt, i):
    #np.savetxt('andmed_'+str(i)+str(dt)+'.txt', (tht, ptht, r, pfii))

# Kontrollimata / pooleli; tulevad faasiportreed
#def draw():
    #return


if __name__ == "__main__":
  

    # Osakeste arv
    Number = 50
    # Nurga algväärtus
    Tht = np.pi/2+1
    # Impulsi algväärtus
    Ptht = 0+5
    # Samm
    Step = np.float32(1/32000)
    
    Blocksize = 512
    Outstep = 10

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

    knl_tht = prg.Theta
    knl_ptht = prg.Ptht


    # Parameetrid
    tht_host = np.full(Number, Tht, dtype=np.float32)
    ptht_host = np.full(Number, Ptht, dtype=np.float32)
    r_host = np.linspace(10, 100, Number).astype(np.float32)
    pfii_host = np.linspace(1, 5, Number).astype(np.float32)


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
    max_steps = 1000

    # Kontrollimata / pooleli
    #saving = input("Save? y/n : ")
    #drawing = input("Graph? y/n : ")

    try:
        while not finish:
            if n % Outstep == 0:
                cl.enqueue_copy(queue, tht_host, tht_dev)
                cl.enqueue_copy(queue, ptht_host, ptht_dev)

                # Kontrollimata / pooleli
                #if saving == "y":
                #    save(tht_host, ptht_host, r_host, pfii_host, Step, n)

                #if drawing == "y":
                #    draw()

                # Protsessi kontrolliks
                norm_val = np.linalg.norm(tht_host)
                print(f"Step {n}: tht_host norm = {norm_val:4f}")
                norm_val2 = np.linalg.norm(ptht_host)
                print(f"Step {n}: ptht_host norm = {norm_val2:4f}")

            knl_tht(queue, gws, lws, tht_dev, ptht_dev, r_dev, Step, np.int32(Number))

            # Kasutab uusi theta väärtusi. Kas peaks kasutama vanu?
            knl_ptht(queue, gws, lws, tht_dev, ptht_dev, r_dev, pfii_dev, Step, np.int32(Number))

            n += 1

            if n >= max_steps:
                # Kontrolliks
                print(tht_host)
                finish = True

    except KeyboardInterrupt:
        print("Interrupted")



tht_dev.release()
ptht_dev.release()


print('end')
