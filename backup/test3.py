def f1(**kwargs):
    print(kwargs)
    f2(**kwargs)
    print()

def f2(**kwargs):
    print(kwargs)
    print()

f1(x=10,y=100)