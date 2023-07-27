get_ipython().system('kg download -u $KG_USER -p $KG_PASSWORD -c dogs-vs-cats-redux-kernels-edition')

get_ipython().system('mkdir -vp data')

get_ipython().system('unzip train.zip')

get_ipython().system('unzip test.zip')

get_ipython().system('mv train data')
get_ipython().system('mv test data')

get_ipython().system('mv train.zip data')
get_ipython().system('mv test.zip data')

get_ipython().system('mkdir -vp data/train/dog')
get_ipython().system('mkdir -vp data/train/cat')
get_ipython().system('mkdir -vp data/valid/dog')
get_ipython().system('mkdir -vp data/valid/cat')
get_ipython().system('mkdir -vp data/test/unknown')

get_ipython().system('find data/train -type d -name "dog" -prune -o -name \'dog*.jpg\' | xargs -I {} mv {} data/train/dog')

get_ipython().system('find data/train -type d -name "cat" -prune -o -name \'cat*.jpg\' | xargs -I {} mv {} data/train/cat')

get_ipython().system('mv data/train/dog/dog.5*.jpg data/valid/dog')

get_ipython().system('mv data/train/cat/cat.5*.jpg data/valid/cat')

get_ipython().system('mv data/test/ data/test/unknown')

get_ipython().system('find data/test -type d -name "unknown" -prune -o -type f | xargs -I {} mv {} data/test/unknown')

get_ipython().system('mv sample_submission.csv data')



