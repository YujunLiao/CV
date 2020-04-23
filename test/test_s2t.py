from dl.utils.s2t import ms2st, ss2st


def test_ms2st():
    s2ts = ms2st(['d1', 'd2', 'd3'], ['d2', 'd3'])
    assert len(s2ts) == 2
    print(s2ts)

def test_ss2st():
    s2ts = ss2st(['d1', 'd2', 'd3'], ['d2', 'd3'])
    assert len(s2ts) == 4
    print(s2ts)