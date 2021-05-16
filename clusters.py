import numpy as np
import torch
import os
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn import preprocessing
from sklearn.datasets import make_blobs
import math
PATH = 'criteria.pth'
criteria_num = 3
num_t = 1024

dataset_path = '/Users/a18651298/Desktop/data/'

# function for loading data


def load_dataset():
    data = np.zeros(shape=(1099, 40, 1024))
    exclude_list = ['src', '.DS_Store', '.ipynb_checkpoints']
    dir_list = os.listdir(dataset_path)
    dir_list = [i for i in dir_list if i not in exclude_list]
    count = 0
    for file_name in dir_list:
        data[count] = torch.load(os.path.join(dataset_path, file_name)).numpy()
        count += 1
    return data


# functions for filtering outliers


def iqr_filter(values_list, strict=False):
    if strict:
        koef = 1.5
    else:
        koef = 3
    result = list(values_list)
    q_1 = np.percentile(values_list, 25)
    q_3 = np.percentile(values_list, 75)
    iqr = q_3 - q_1
    x_down = q_1 - koef * iqr
    x_up = q_3 + koef * iqr
    for value in result:
        if value > x_up or value < x_down:
            result.remove(value)
    return np.array(result)


def filter_and_mean(tensor, strict_filtering=False):
    tensor = abs(np.fft.fft(tensor, axis=1))
    mean_values = np.zeros(shape=tensor.shape[1])
    for coord_num in range(tensor.shape[1]):
        values = iqr_filter(tensor[:, coord_num], strict=strict_filtering)
        mean_values[coord_num] = np.mean(values)
    return mean_values


def find_outliers_and_means(word_dataset, strict_filter=False):
    num_word = word_dataset.shape[0]
    num_time = word_dataset.shape[2]
    processed_dataset = np.zeros(shape=(num_word, num_time))
    print('Getting rid of outliers:')
    for tensor_number in range(num_word):
        print(str(tensor_number + 1) + ' / ' + str(num_word))
        processed_dataset[tensor_number] = filter_and_mean(word_dataset[tensor_number], strict_filtering=strict_filter)
    return processed_dataset


# function forming the results of clustering


def form_cluster_members(data, distr, number_of_clusters):
    result = []
    for cluster_number in range(number_of_clusters):
        cluster_power = list(distr).count(cluster_number)  # number of elements in cluster
        res = np.zeros(shape=(cluster_power, num_t))
        count = 0
        for index in range(len(distr)):
            if distr[index] == cluster_number:
                res[count] = data[index]
                count += 1
        result.append(res)
    return result  # list of numpy arrays of different sizes


# functions for calculating Davies-Bouldin Index


def s_function(cluster_members, centers, cluster_number):
    s_value = 0
    cluster_center = centers[cluster_number]
    cluster_data = cluster_members[cluster_number]
    for word in range(cluster_data.shape[0]):
        s_value += np.linalg.norm(cluster_data[word] - cluster_center)
    s_value /= cluster_data.shape[0]
    return s_value


def get_s_values(cluster_members, centers):
    s_values = []
    for j in range(len(centers)):
        s_values.append(s_function(cluster_members, centers, j))
    return s_values


def db_index(centers, cluster_members):
    dbi = 0
    num_clusters = len(centers)
    s_values = get_s_values(cluster_members, centers)
    for cluster_number in range(num_clusters):
        max_value = 0
        for other_cluster in range(num_clusters):
            if other_cluster != cluster_number:
                value = (s_values[cluster_number] + s_values[other_cluster]) / np.linalg.norm(centers[cluster_number]
                                                                                              - centers[other_cluster])
                if value > max_value:
                    max_value = value
        dbi += max_value
    dbi /= len(centers)
    return dbi


# functions for calculating Score function


def find_bcd_value(data, centers, distr, cluster_members, needs_division=True):
    num_clusters = len(centers)
    num_data = len(distr)
    mean_value = np.mean(data, axis=0)
    sum_value = 0
    for cluster_num in range(num_clusters):
        sum_value += cluster_members[cluster_num].shape[0] * np.linalg.norm(centers[cluster_num] - mean_value)
    if needs_division:
        sum_value /= (num_data * num_clusters)
    return sum_value


def find_wcd_value(centers, cluster_members, needs_division=True):
    num_clusters = len(centers)
    sum_value = 0
    for cluster_num in range(num_clusters):
        subsum_value = 0
        for word_num in range(cluster_members[cluster_num].shape[0]):
            subsum_value += np.linalg.norm(cluster_members[cluster_num][word_num] - centers[cluster_num])
        if needs_division:
            subsum_value /= cluster_members[cluster_num].shape[0]
        sum_value += subsum_value
    return sum_value


def score_function(data, centers, distr, cluster_members):
    bcd_value = find_bcd_value(data, centers, distr, cluster_members)
    wcd_value = find_wcd_value(centers, cluster_members)
    score = 1 - 1 / (math.exp(math.exp(bcd_value - wcd_value)))
    return score


# function for calculating Calinski–Harabasz Index


def ch_index(data, centers, distr, cluster_members):
    num_clusters = len(centers)
    num_data = len(distr)
    upper = find_bcd_value(data, centers, distr, cluster_members, needs_division=False)
    lower = find_wcd_value(centers, cluster_members, needs_division=False)
    ch = (num_data - num_clusters) * (upper / lower) / (num_clusters - 1)
    return ch


# function for calculating COP Index


def cop_index(data, centers, distr, cluster_members):
    num_clusters = len(centers)
    sum_value = 0
    for cluster_num in range(num_clusters):
        upper = s_function(cluster_members, centers, cluster_num)
        min_of_maxes = 0
        counter = 0
        for word_num in range(data.shape[0]):
            if distr[word_num] != cluster_num:
                max_dist = 0
                for tensor_num in range(cluster_members[cluster_num].shape[0]):
                    distance = np.linalg.norm(data[word_num] - cluster_members[cluster_num][tensor_num])
                    if distance > max_dist:
                        max_dist = distance
                counter += 1
                if max_dist < min_of_maxes or counter == 1:
                    min_of_maxes = max_dist
        sum_value += cluster_members[cluster_num].shape[0] * upper / min_of_maxes
    sum_value /= data.shape[0]
    return sum_value


# functions for calculating Dunn Index


def get_delta_value(centers, cluster_members, number_k, number_l):
    sum_k = 0
    sum_l = 0
    for k_members in range(cluster_members[number_k].shape[0]):
        sum_k += np.linalg.norm(cluster_members[number_k][k_members] - centers[number_k])
    for l_members in range(cluster_members[number_l].shape[0]):
        sum_l += np.linalg.norm(cluster_members[number_l][l_members] - centers[number_l])
    delta = (sum_k + sum_l) / (cluster_members[number_k].shape[0] + cluster_members[number_l].shape[0])
    return delta


def get_nabla_value(centers, cluster_members, number_k):
    sum_k = 0
    for k_members in range(cluster_members[number_k].shape[0]):
        sum_k += np.linalg.norm(cluster_members[number_k][k_members] - centers[number_k])
    nabla = (2 * sum_k) / cluster_members[number_k].shape[0]
    return nabla


def dunn_index(centers, cluster_members):
    num_clusters = len(centers)
    min_min_delta = 0
    max_nabla = 0
    big_counter = 0
    for cluster_num in range(num_clusters):
        counter = 0
        min_delta = 0
        for other_cluster in range(num_clusters):
            if other_cluster != cluster_num:
                delta = get_delta_value(centers, cluster_members, cluster_num, other_cluster)
                counter += 1
                if delta < min_delta or counter == 1:
                    min_delta = delta
        big_counter += 1
        if min_delta < min_min_delta or big_counter == 1:
            min_min_delta = min_delta

        nabla = get_nabla_value(centers, cluster_members, cluster_num)
        if nabla > max_nabla:
            max_nabla = nabla
    return min_min_delta / max_nabla


# evaluating the quality of clusterization with 4 criteria


def eval_quality(data, centers, distr):
    num_clusters = len(centers)
    cluster_members = form_cluster_members(data, distr, num_clusters)
    quality = np.zeros(shape=criteria_num)
    quality[0] = davies_bouldin_score(data, distr)
    quality[1] = calinski_harabasz_score(data, distr)
    quality[2] = silhouette_score(data, distr)
    # first two should be minimized, latter two - maximized
    return quality


# launching k-means clusterization


def launch_k_means(data, k_clusters, num_experiments):
    results = np.zeros(shape=(num_experiments, criteria_num))
    print('Clusterizing and evaluating criteria for k = ' + str(k_clusters) + '...')
    for experiment_num in range(num_experiments):
        k_means = KMeans(n_clusters=k_clusters)
        k_means.fit(data)
        #results[experiment_num] = eval_quality(data, k_means.cluster_centers_, k_means.fit_predict(data))
        print(calinski_harabasz_score(data, k_means.fit_predict(data)))
    mean_results = np.mean(results, axis=0)
    return mean_results


def choose_best_k(matrix):
    # we have (maxK - minK) x (criteria_number) matrix
    # let's normalize every criterion
    min_max_scaler = preprocessing.MinMaxScaler()
    for criterion in range(matrix.shape[1]):
        vector = min_max_scaler.fit_transform(np.reshape(matrix[:, criterion], newshape=(-1, 1)))
        matrix[:, criterion] = np.reshape(vector, newshape=-1)
    # now choose the closest vector to (0, 0, 1, 1)
    best_k = 0
    ideal_vector = np.array([0, 0, 1, 1])
    min_distance = 2
    for k_clusters in range(matrix.shape[0]):
        distance = np.linalg.norm(matrix[k_clusters] - ideal_vector)
        if distance < min_distance:
            min_distance = distance
            best_k = k_clusters
    return best_k


if __name__ == "__main__":
    print("Loading data ...")
    # dataset = np.random.uniform(1, 10, (1000, 40, 1024))
    #dataset = load_dataset()
    datas = make_blobs(n_samples=400, n_features=1024, centers=4, cluster_std=1.6,
                      random_state=50)  # create np array for data points
    dataset = datas[0]
    #processed_data = find_outliers_and_means(dataset)
    experimental_results = np.zeros(shape=(8-2, criteria_num))
    for k in range(2, 8):
        experimental_results[k-2] = launch_k_means(dataset, k, 20)
    print('Matrix with the results of experiments (rows stand for K in k-means method (k varies from 2 to 8)' +
          ', columns stand for criteria values (DBI, COP, CH, DUNN respectively):')
    print(experimental_results)
    torch.save(torch.from_numpy(experimental_results), PATH)
    best_num_of_clusters = choose_best_k(experimental_results) + 2
    print('Recommended number of clusters is: ' + str(best_num_of_clusters))