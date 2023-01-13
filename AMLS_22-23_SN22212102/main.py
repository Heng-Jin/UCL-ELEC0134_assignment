import A1.A1 as A1
import A2.A2 as A2
import B1.B1 as B1
import B2.B2 as B2
import time

def main():
    A1.run()
    time.sleep(5)
    A2.run()
    time.sleep(5)
    B1.run()
    time.sleep(5)
    B2.run()
    time.sleep(5)

if __name__ == '__main__':
    main()