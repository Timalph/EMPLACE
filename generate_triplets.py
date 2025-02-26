import os
import pandas as pd
from datetime import timedelta
from datetime import datetime
from time import time
import argparse
from os import system
import sys
cluster_content = [ '20160509TMX7315120208-000040pano_0000_000841_4.96448125072271_52.4063760039986.',
					'20181219TMX7316010203-001026pano_0009_000016_4.96447563315362_52.4063758743326.jpg',
					'20190701TMX7316010203-001262pano_0001_000357_4.96449594736324_52.4063712245894.jpg',
					'20200622TMX7316010203-001752pano_0000_001689_4.96448843742371_52.4063737320951.jpg',
					'20211227TMX7316010203-002352pano_0001_000025_4.96449702995053_52.4063708196698.jpg',
					'20211227TMX7316010203-002352pano_0003_000045_4.96449855828846_52.4063702783855.jpg',
					'20220504TMX7316010203-002560pano_0005_000293_4.96448233788673_52.4063733633925.jpg',
					'20221223TMX7316010203-002888pano_0004_000068_4.96447710832335_52.406373093104.jpg']

def path2date(path):
	year = int(path[:4])
	month = int(path[4:6])
	day = int(path[6:8])
	if month > 12:
		month = month%12
		year += 1
	date = datetime(year, month, day)

	return date

def str2date(s):
	year, month, day = map(int, s.split('-'))
	if month > 12:
		month = month%12
		year += 1
	date = datetime(year, month, day)
	return date

def generate_permutations(cluster_content, cluster_path = '', distances=False):
	""" Function that generates all possible permuations of anchors, positives, and negatives. 

	Currently the only implementation allows to generate triplets where 
	the distance between anchor and positive is smaller than the distance between anchor and negative. 
	This is single way only, so no anchor in the middle and positive and negative on either side.

	distances is a list of [pos_d, neg_d], which specify the maximum pos distance, and minimum neg distance.

	"""
	datelist = [path2date(x) for x in cluster_content]
	date_dict = {key: val for key, val in zip(datelist, cluster_content)}

	### Generate permutations where the distance of Anchor to Positive is smaller than the distance of Anchor to Negative. 

	column_names = ['anchor_dt', 'positive_dt', 'negative_dt', 'anchor_path', 'positive_path', 'negative_path']

	df_of_permutations = pd.DataFrame(columns = column_names)

	df_of_complement_permutations = pd.DataFrame(columns = column_names)
	for idx, (anchor, anchor_path) in enumerate(zip(datelist, cluster_content)):

		### For each file, make a list of potential candidates for pos/neg.
		### These are all images taken after the anchor. 

		candidates = datelist[idx+1:]
		candidates_path = cluster_content[idx+1:]
		for c_idx, (positive, positive_path) in enumerate(zip(candidates, candidates_path)):

			### Same applies here, but now for positive and negative images. 

			negatives = candidates[c_idx+1:]
			negatives_path = candidates_path[c_idx+1:]
			for negative, negative_path in zip(negatives, negatives_path):

				## if the pos is closer to the anchor than the neg you can use it. 
				#print('apn', anchor, positive, negative)
				if (positive - anchor) < (negative - anchor):
					new_row_dict = {key: [val] for key, val in zip(column_names, [anchor, positive, 
						negative, cluster_path + anchor_path, cluster_path + positive_path, cluster_path + negative_path])}

					new_row = pd.DataFrame(new_row_dict)
					df_of_permutations = pd.concat([df_of_permutations, new_row], ignore_index=True)
#					df_of_permutations = df_of_permutations.append(new_row).reset_index(drop=True)

				if (negative - positive) < (negative - anchor):
					complement_row = {key: [val] for key, val in zip(column_names, [negative, positive, 
						anchor, cluster_path + negative_path, cluster_path + positive_path, cluster_path + anchor_path])}



					new_complement_row = pd.DataFrame(complement_row)
					df_of_complement_permutations = pd.concat([df_of_complement_permutations, new_row], ignore_index=True)
#					df_of_complement_permutations = df_of_complement_permutations.append(new_complement_row).reset_index(drop=True)


	### Apply constraints
	if distances:
		df_of_permutations = filter_distances(df_of_permutations, distances[0], distances[1])
		df_of_complement_permutations = filter_distances(df_of_complement_permutations, distances[0], distances[1], backwards=True)


	return df_of_permutations, df_of_complement_permutations


def filter_distances(df, pos, neg, backwards=False, ):
	"""
	Filter distances takes a dataframe and a pos and neg threshold and drops 
	all rows where the distance from pos to anchor is more than pos.
	Vice versa for the rows where distance from neg to anchor is less than neg. 

	Backwards indicates the df is generated backwards through time. 
	"""
	pos = timedelta(pos)
	neg = timedelta(neg)
	todrop = []

	if not backwards:
		for index in range(len(df)):
			if abs(df.iloc[index]['positive_dt'] - df.iloc[index]['anchor_dt']) > pos:
				todrop.append(index)
			elif abs(df.iloc[index]['negative_dt'] - df.iloc[index]['anchor_dt']) < neg:
				todrop.append(index)

	elif backwards:
		for index in range(len(df)):
			if (df.iloc[index]['anchor_dt'] - df.iloc[index]['positive_dt']) > pos:
				todrop.append(index)
			elif (df.iloc[index]['anchor_dt'] - df.iloc[index]['negative_dt']) < neg:
				todrop.append(index)

	filtered_df = df.drop(index = todrop)
	filtered_df = filtered.drop(columns='Unnamed: 0').reset_index(drop=True)
	return filtered_df

def filter_distances_after_saving(df, pos, neg, pos_prox=0, neg_prox=9999, step=1000000000):
	pos = timedelta(pos)
	neg = timedelta(neg)
	pos_prox = timedelta(pos_prox)
	todrop = []

	if neg_prox == 9999:
		length = len(df)
		for index in range(length):
			posdt = str2date(df.iloc[index]['positive_dt'])
			anchordt = str2date(df.iloc[index]['anchor_dt'])
			if (abs(posdt - anchordt) < pos_prox) or (abs(posdt - anchordt) > pos) or (abs(str2date(df.iloc[index]['negative_dt']) - anchordt < neg)):
				todrop.append(index)
			if index%step==0:
				print('{}/{}'.format(index,length))
		filtered_df = df.drop(index = todrop)


	elif neg_prox != 9999:
		neg_prox = timedelta(neg_prox)
		length = len(df)
		for index in range(length):
			posdt = str2date(df.iloc[index]['positive_dt'])
			anchordt = str2date(df.iloc[index]['anchor_dt'])
			negativedt = str2date(df.iloc[index]['negative_dt'])

			if (abs(posdt - anchordt) < pos_prox) or (abs(posdt - anchordt) > pos) or (abs(negativedt - anchordt < neg)) or (abs(negativedt - anchordt > neg_prox)):
				todrop.append(index)
			if index%step==0:
				print('{}/{}'.format(index,length))
		filtered_df = df.drop(index = todrop)


	return filtered_df
#### 
### Function to filter the large DFs to be temporally congruent through time e.q. only going one way anc < pos < neg.
def filter_temporal_congruent(df, step=100000):
	todrop=[]
	length = len(df)
	for index in range(length):
		neg = str2date(df.iloc[index]['negative_dt'])
		pos = str2date(df.iloc[index]['positive_dt'])
		anc = str2date(df.iloc[index]['anchor_dt'])
		# Check if we're going forward in time
		if not ((anc < pos < neg) or (anc > pos > neg)):
			todrop.append(index)
		if index%step==0:
			print('{}/{}'.format(index,length))
	return df.drop(index=todrop)

def check_tcg(df):
	length = len(df)
	for index in range(length):
		neg = str2date(df.iloc[index]['negative_dt'])
		pos = str2date(df.iloc[index]['positive_dt'])
		anc = str2date(df.iloc[index]['anchor_dt'])
		if not ((anc < pos < neg) or (anc > pos > neg)):
			print(neg, pos, anc)
			break


def unittests(df, cdf):
	start = time()
	assert sum(df.duplicated()) == 0

	# assert the distance between anchor and pos is smaller than anchor and neg.
	for i in range(len(df)):
		
		assert (df.iloc[i]['positive_dt'] - df.iloc[i]['anchor_dt']) < (df.iloc[i]['negative_dt'] - df.iloc[i]['anchor_dt'])
		assert path2date(df.iloc[i]['negative_path']) == df.iloc[i]['negative_dt']
		assert path2date(df.iloc[i]['positive_path']) == df.iloc[i]['positive_dt']
		assert path2date(df.iloc[i]['anchor_path']) == df.iloc[i]['anchor_dt']

	for i in range(len(cdf)):
		assert (cdf.iloc[i]['anchor_dt'] - cdf.iloc[i]['positive_dt']) < (cdf.iloc[i]['anchor_dt'] - cdf.iloc[i]['negative_dt'])


	print('All tests passed in {}'.format(time() - start))

def test():

	start = time()
	perms, complement_perms = generate_permutations(cluster_content)
	print(perms)
	print(time() - start)
	#print(perms[['anchor_dt', 'positive_dt', 'negative_dt']])
	#print(complement_perms[['anchor_dt', 'positive_dt', 'negative_dt']])


	#print(perms)
	#print(complement_perms)
	unittests(perms, complement_perms)

	start = time()
	perms, complement_perms = generate_permutations(cluster_content, distances = [366, 750])
	print(time() - start)
	#print(perms, complement_perms)
	#print(perms)
	#print(complement_perms)
	unittests(perms, complement_perms)

	#print(perms[['anchor_dt', 'positive_dt', 'negative_dt']])
	#print(complement_perms[['anchor_dt', 'positive_dt', 'negative_dt']])
	perms['positive_path'] = 'Zunderdorop/12/' + perms['positive_path']
	print(perms)


def main(args):
	path = args.path
	column_names = ['anchor_dt', 'positive_dt', 'negative_dt', 'anchor_path', 'positive_path', 'negative_path', 'cluster', 'nb']
	all_triplets = pd.DataFrame(columns=column_names)

	no_of_nbs = 0
	for d in os.listdir(path):
		if os.path.isdir(os.path.join(path, d)):
			no_of_nbs += 1


	for nb_idx, nb in enumerate(os.listdir(path)):
		if os.path.isdir(os.path.join(path, nb)):
			nb_df = pd.DataFrame(columns=column_names[:-1])
			nb_path = os.path.join(path,nb)
			for cluster_idx, cluster in enumerate(os.listdir(nb_path)):

				cluster_df = pd.DataFrame(columns=column_names[:-2])
				cluster_path = os.path.join(nb_path, cluster)
				cluster_content = os.listdir(cluster_path)
				df, cdf = generate_permutations(cluster_content)
				cluster_df = cluster_df.append(df)
				cluster_df = cluster_df.append(cdf)
				cluster_df['cluster'] = cluster
				nb_df = nb_df.append(cluster_df)
				_ = system('clear')
				print('{} {}/{}'.format(nb, nb_idx, no_of_nbs))
				print('{}/{}'.format(cluster_idx, len(os.listdir(nb_path))))

				if args.minitest:
					break
			nb_df['nb'] = nb
			all_triplets = all_triplets.append(nb_df)

	all_triplets.to_csv(path[:-1] + '_allclusters.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/media/tim/D/TimeMachineBMVC/1/')
    parser.add_argument('--minitest', type=bool, default=False)
    args = parser.parse_args()
    
    main(args)


