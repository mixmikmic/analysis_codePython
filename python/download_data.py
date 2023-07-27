get_ipython().system('kg download -u $KG_USER -p $KG_PASSWORD -c dogs-vs-cats-redux-kernels-edition')

get_ipython().system('mkdir -vp data/all')

get_ipython().system('mv *.zip data')
get_ipython().system('mv *.csv data')

get_ipython().system('cd data && unzip train.zip')

get_ipython().system('cd data && unzip test.zip')



