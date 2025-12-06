
def max_voltage(nums):
    tot = 0
    for num in nums:
        tot += max_bar(num)
    return tot


def max_bar(nums):
    l = 0
    max_number = int(nums[0])
    for r in range(1, len(nums)):
        f_dig = int(nums[l])
        s_dig = int(nums[r])
        num = f_dig*10 + s_dig
        max_number = max(max_number, num)
        if int(nums[r]) > int(nums[l]):
            l = r
    return max_number


with open('day3.txt') as f:
    lines = f.readlines()

nums = [line.strip() for line in lines]
test = ['987654321111111']
tot = max_voltage(nums)
print(tot)
