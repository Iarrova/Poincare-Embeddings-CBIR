granularity_3_cifar10 = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
granularity_2_cifar10 = [[0], [1, 9], [2], [3, 5], [4, 7], [6], [8]]
granularity_1_cifar10 = [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]

cifar10_hierarchy = [granularity_1_cifar10, granularity_2_cifar10, granularity_3_cifar10]

granularity_5_cifar100 = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19],
    [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39],
    [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],
    [60], [61], [62], [63], [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79],
    [80], [81], [82], [83], [84], [85], [86], [87], [88], [89], [90], [91], [92], [93], [94], [95], [96], [97], [98], [99]]

granularity_4_cifar100 = [[0, 51, 53, 57, 83], [54, 62, 70, 82, 92], [47, 52, 56, 59, 96], [22, 39, 40, 86, 87],
    [5, 20, 25, 84, 94], [9, 10, 16, 28, 61], [8, 13, 48, 58, 90], [41, 69, 81, 85, 89],
    [12, 17, 37, 68, 76], [23, 33, 49, 60, 71], [3, 42, 43, 88, 97], [15, 19, 21, 31, 38],
    [34, 63, 64, 66, 75], [36, 50, 65, 74, 80], [2, 11, 35, 46, 98], [4, 30, 55, 72, 95],
    [1, 32, 67, 73, 91], [26, 45, 77, 79, 99], [27, 29, 44, 78, 93], [6, 7, 14, 18, 24]]

granularity_3_cifar100 = [[0, 51, 53, 57, 83, 54, 62, 70, 82, 92, 47, 52, 56, 59, 96],
    [22, 39, 40, 86, 87, 5, 20, 25, 84, 94, 9, 10, 16, 28, 61],
    [8, 13, 48, 58, 90, 41, 69, 81, 85, 89], [12, 17, 37, 68, 76], [23, 33, 49, 60, 71],
    [3, 42, 43, 88, 97, 15, 19, 21, 31, 38, 34, 63, 64, 66, 75, 36, 50, 65, 74, 80, 2, 11, 35, 46, 98],
    [4, 30, 55, 72, 95, 1, 32, 67, 73, 91], [26, 45, 77, 79, 99, 27, 29, 44, 78, 93, 6, 7, 14, 18, 24]]

granularity_2_cifar100 = [[0, 51, 53, 57, 83, 54, 62, 70, 82, 92, 47, 52, 56, 59, 96],
    [22, 39, 40, 86, 87, 5, 20, 25, 84, 94, 9, 10, 16, 28, 61, 8, 13, 48, 58, 90, 41, 69, 81, 85, 89, 12, 17, 37, 68, 76],
    [23, 33, 49, 60, 71],
    [3, 42, 43, 88, 97, 15, 19, 21, 31, 38, 34, 63, 64, 66, 75, 36, 50, 65, 74, 80, 2, 11, 35, 46, 98, 4, 30, 55, 72, 95, 1, 32, 67, 73, 91, 26, 45, 77, 79, 99, 27, 29, 44, 78, 93, 6, 7, 14, 18, 24]]

granularity_1_cifar100 = [[0, 51, 53, 57, 83, 54, 62, 70, 82, 92, 47, 52, 56, 59, 96, 3, 42, 43, 88, 97, 15, 19, 21, 31, 38, 34, 63, 64, 66, 75, 36, 50, 65, 74, 80, 2, 11, 35, 46, 98, 4, 30, 55, 72, 95, 1, 32, 67, 73, 91, 26, 45, 77, 79, 99, 27, 29, 44, 78, 93, 6, 7, 14, 18, 24],
    [22, 39, 40, 86, 87, 5, 20, 25, 84, 94, 9, 10, 16, 28, 61, 8, 13, 48, 58, 90, 41, 69, 81, 85, 89, 12, 17, 37, 68, 76, 23, 33, 49, 60, 71]]

cifar100_hierarchy = [granularity_1_cifar100, granularity_2_cifar100, granularity_3_cifar100, granularity_4_cifar100, granularity_5_cifar100]


def get_similar_classes(label, granularity, dataset):
    if dataset == 'CIFAR10':
        hierarchy_level = cifar10_hierarchy[granularity - 1]
    elif dataset == 'CIFAR100':
        hierarchy_level = cifar100_hierarchy[granularity - 1]

    for lst in hierarchy_level:
        if label in lst:
            return lst