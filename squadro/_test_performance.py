from time import time

if __name__ == "__main__":
    t = time()
    
    a = 0
    for i in range(100000000):
        a = a + 1
    
    print (time() - t)
