from collections import defaultdict
nums = [1, 1, 1, 2, 2, 3]
nums2 = [1]


def kfrequent(nums, k):
    result = []
    num_map = defaultdict(int)
    buckets = [[] for i in range(len(nums))]
    for number in nums:
        num_map[number] += 1

    for key, value in num_map.items():
        buckets[value].append(key)

    for bucket in buckets[::-1]:
        for number in bucket:
            result.append(number)
            if len(result) == k:
                return result


print(kfrequent(nums2, 2))
