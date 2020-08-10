srun --mem=32G -p cpu python reddit/r_preprocessing.py reddit/data/orig/2014_gendered.csv reddit/data/labeled/2014_gendered_labeled.csv reddit/labelgen_models/0810_labelgen.pkl
echo first done
srun --mem=32G -p cpu python reddit/r_preprocessing.py reddit/data/orig/2014b.csv reddit/data/labeled/2014b_labeled.csv reddit/labelgen_models/0810_labelgen.pkl
