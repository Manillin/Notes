def func(s):
    n = len(s)
    dupe = s + s
    for i in range(1, len(dupe)):
        sstring = dupe[i:i+n]
        print(sstring)
        if sstring == s:
            if i < n:
                return True
            else:
                return False
    return False


s = 'abcabc'
func(s)
