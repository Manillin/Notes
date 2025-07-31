def threeSum(nums):
    nums.sort()
    triplets = []

    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue

        l, r = i+1, len(nums)-1
        while l < r:
            threeSum = nums[i] + nums[l] + nums[r]
            if threeSum > 0:
                r -= 1
            elif threeSum < 0:
                l += 1
            else:
                triplets.append([nums[i], nums[l], nums[r]])
                l += 1
                while l < r and nums[l] == nums[l-1]:
                    l += 1
    return triplets


def twoSum(numbers, target):
    l, r = 0, len(numbers)-1
    while l < r:
        twosum = numbers[l] + numbers[r]
        if twosum > target:
            r -= 1
        elif twosum < target:
            l += 1
        else:
            return [l+1, r+1]
