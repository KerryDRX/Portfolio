def uncertainty(alpha):
    pred = alpha.argmax(dim=1)
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
        
    least_confidence = 1 - p.max(dim=1)[0]
        
    p_top2 = p.topk(k=2, dim=1)[0]
    margin_of_confidence = p_top2[:, 0] - p_top2[:, 1]
    ratio_of_confidence = p_top2[:, 0] / p_top2[:, 1]
        
    entropy = - (p * (p + 1e-7).log()).sum(dim=1)
    data = (p * ((S + 1).digamma() - (alpha + 1).digamma())).sum(dim=1)
    distributional = entropy - data
        
    variance = (p - p ** 2).sum(dim=1)
    epistemic = variance / (S[:, 0] + 1)
    aleatoric = variance - epistemic
        
    differential_entropy = alpha.lgamma().sum(dim=1) - S[:, 0].lgamma() - (
        (alpha - 1) * (alpha.digamma() - S.digamma())
    ).sum(dim=1)
        
    return {
        'p': p,
        'pred': pred,
        'least_confidence': least_confidence,
        'margin_of_confidence': margin_of_confidence,
        'ratio_of_confidence': ratio_of_confidence,
        'entropy': entropy,
        'data': data,
        'distributional': distributional,
        'variance': variance,
        'aleatoric': aleatoric,
        'epistemic': epistemic,
        'differential_entropy': differential_entropy,
    }