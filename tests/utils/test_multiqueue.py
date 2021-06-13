from lwsspy.utils.multiqueue import multiwrapper


def processing_function(*args, **kwargs):
    print("Hello from ", *args)
    return args[0] ** 2


def test_run_multiqueue():

    N = 10

    args = [(i, ) for i in range(N)]
    kwargs = [dict(i=1, j=2) for _ in range(N)]

    results = multiwrapper(processing_function, args, kwargs)

    for result in results:
        print(result)


if __name__ == "__main__":
    print("Hello")
    test_run_multiqueue()
    print("Hello end")
