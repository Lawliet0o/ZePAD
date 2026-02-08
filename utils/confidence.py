import torch
import torch.nn.functional as F

def self_confidence(x,k=2):
    
    return x * torch.exp(k * (x - 0.5))
    # return x
# 2. Entropy-based confidence
def entropy_confidence(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


# 3. Temperature-scaling confidence
def temperature_confidence(logits, T=1.0, k=3):
    p = F.softmax(logits / T, dim=-1)
    m = p.max()
    return m * torch.exp(k * (m - 0.5))

def federal_with_zepad(logits1, logits2, logits3):
    victim_probabilities = torch.softmax(logits1, dim=1)
    helper1_probabilities = torch.softmax(logits2, dim=1)
    helper2_probabilities = torch.softmax(logits3, dim=1)

    max_val_vic = torch.max(victim_probabilities, dim=1)[0]  #  (batch_size,)
    max_val_help1 = torch.max(helper1_probabilities, dim=1)[0] #  (batch_size,)
    max_val_help2 = torch.max(helper2_probabilities, dim=1)[0] #  (batch_size,)
    # calculate confidence value
    k = 3 # confidence value function parameter
    vitctim_confidence = self_confidence(max_val_vic,k)
    helper1_confidence = self_confidence(max_val_help1,k)
    helper2_confidence = self_confidence(max_val_help2,k)

    # get weight
    sum = vitctim_confidence + helper1_confidence + helper2_confidence
    vic_weight = vitctim_confidence / sum
    helper1_weight = helper1_confidence / sum
    helper2_weight = helper2_confidence / sum

    vic_weight = vic_weight.unsqueeze(1)  #  (batch_size, 1)
    helper1_weight = helper1_weight.unsqueeze(1)  #  (batch_size, 1)
    helper2_weight = helper2_weight.unsqueeze(1)  #  (batch_size, 1)

    return vic_weight*logits1 + helper1_weight*logits2 + helper2_weight*logits3

def federal_with_equal(logits1, logits2, logits3):
    return (logits1+logits2+logits3)/3

def federal_with_entropy(logits1, logits2, logits3):

    confidence1=entropy_confidence(logits1)
    confidence2=entropy_confidence(logits2)
    confidence3=entropy_confidence(logits3)

    w1 = confidence1 / (confidence1+confidence2+confidence3)

    w1 = w1.unsqueeze(1)
    w2 = confidence2 / (confidence1+confidence2+confidence3)
    w2 = w2.unsqueeze(1)
    w3 = confidence3 / (confidence1+confidence2+confidence3)
    w3 = w3.unsqueeze(1)

    return w1*logits1+w2*logits2+w3*logits3