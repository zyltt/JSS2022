# coding=utf-8

def Print_saitama(g,size):
    for k in range(size):
        for q in range(size):
            print(g.Map[k][q].level, ':', g.Map[k][q].m," ")
        print("")
    return