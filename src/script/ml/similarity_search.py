import torch

def search(Dx, Dy, tx, seq):
    for _ in tx:
        dis = torch.norm(Dx-_, dim=1)
        tab = dis.sort(descending=False)[1]
        ind = tab[0]
        if len(seq)==0:
            seq.append(ind)
        else:
            i = 1
            while ind in seq:
                ind = tab[i]
                i+=1
            seq.append(ind)
    x_set = Dx[torch.stack(seq,0)]
    y_set = Dy[torch.stack(seq,0)]
    return x_set, y_set, seq