def subset_sum(numbers, target, partial=[], partial_sum=0):
    if partial_sum <= target:
        yield partial
    if partial_sum >= target:
        return
    for i, n in enumerate(numbers):
        remaining = numbers[i + 1:]
        yield from subset_sum(remaining, target, partial + [n], partial_sum + n)

bets = []

for bet in list(subset_sum([5,10,20,50,100],250)):
    bets.append(bet)

print(len(bets))