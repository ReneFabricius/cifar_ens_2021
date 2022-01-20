import torch
from timeit import default_timer as timer
import sys

from utils import print_memory_statistics


def test_transfer():
    print("Performing transfer test")
    size = (100, 100)
    number = 10000
    dev = torch.device("cuda")
    dtp = torch.float64

    print("Warming up cuda")
    m = torch.tensor([2, 5, 1]).cuda()
    n = torch.tensor([7.3, 5, -1.325879]).cuda()
    a = m + n

    print("Testing to() method")
    start_to = timer()
    for i in range(number):
        t = torch.rand(size).to(device=dev, dtype=dtp)
    end_to = timer()
    print("Method to() finished in {} s".format(end_to - start_to))

    print("Testing cuda() method")
    start_cuda = timer()
    for i in range(number):
        t = torch.rand(size).to(dtype=dtp).cuda()
    end_cuda = timer()
    print("Method cuda() finished in {} s".format(end_cuda - start_cuda))

    print("Testing direct method")
    start_dir = timer()
    for i in range(number):
        t = torch.rand(size, device=dev, dtype=dtp)
    end_dir = timer()
    print("Method direct finished in {} s".format(end_dir - start_dir))


def test_thresholding():
    print("Performing thresholding test")
    size = (10000, 100, 100)
    number = 1
    dev = torch.device("cuda")
    dtp = torch.float64

    print("Warming up cuda")
    m = torch.tensor([2, 5, 1]).cuda()
    n = torch.tensor([7.3, 5, -1.325879]).cuda()
    a = m + n

    del m, n, a

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    eps = 1e-3
    rnd_t = torch.rand(number, *size, device=dev, dtype=dtp)

    print("Testing mask approach")
    start = timer()
    print_memory_statistics()
    for i in range(number):
        cpy = rnd_t[i]
        cpy[cpy <= eps] = eps
        cpy[cpy >= (1 - eps)] = (1 - eps)
        del cpy
    print_memory_statistics()
    end = timer()
    print("Test finished in {}s".format(end - start))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Testing min/max approach")
    start = timer()
    print_memory_statistics()
    for i in range(number):
        cpy = rnd_t[i]
        cpy = torch.max(cpy, torch.tensor([eps], device=dev, dtype=dtp))
        cpy = torch.min(cpy, torch.tensor([1 - eps], device=dev, dtype=dtp))
        del cpy
    print_memory_statistics()
    end = timer()
    print("Test finished in {}s".format(end - start))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Argument is number of test:\n\t1 for transfer test\n\t2 for thresholding")

    if sys.argv[1] == '1':
        test_transfer()
    elif sys.argv[1] == '2':
        test_thresholding()
