import torch
from timeit import default_timer as timer


def perform_test():
    size = (100,100)
    number = 100
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
    print("Method to() finished in {} s".format(end_cuda - start_cuda))

    print("Testing direct method")
    start_dir = timer()
    for i in range(number):
        t = torch.rand(size, device=dev, dtype=dtp)
    end_dir = timer()
    print("Method direct finished in {} s".format(end_dir - start_dir))


if __name__ == "__main__":
    perform_test()