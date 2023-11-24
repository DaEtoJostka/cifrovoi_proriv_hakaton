import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig


# Замените 'your_file_path' на путь к вашему файлу
file_path = 'train_dataset_hackaton2023_train.gzip'

# Загрузка данных
data = pd.read_parquet(file_path)
data = data.drop('group_name', axis=1)

# Convert startdatetime to numeric type
data['startdatetime'] = pd.to_numeric(data['startdatetime'], errors='coerce')

num_col_names = ['date_diff_post','revenue', 'startdatetime','ownareaall_sqm']
cat_col_names = ['customer_id', 'dish_name', 'format_name']

data_config = DataConfig(
    target=['buy_post'],
    continuous_cols=num_col_names, 
    categorical_cols=cat_col_names,
)
trainer_config = TrainerConfig(
    auto_lr_find=True,
    batch_size=1024,
    max_epochs=30,
    accelerator="auto",
)
optimizer_config = OptimizerConfig()


head_config = LinearHeadConfig(
    layers="", 
    initialization="kaiming"
).__dict__ 

model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="32-16", # Number of nodes in each layer
    activation="LeakyReLU", # Activation between each layers
    dropout=0.1,
    initialization="kaiming",
    head = "LinearHead", #Linear Head
    head_config = head_config, # Linear Head Config
    learning_rate = 1e-3
)


tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

train, val = train_test_split(data, test_size=0.2)

tabular_model.fit(train=train, validation=val)

result = tabular_model.evaluate(val)

tabular_model.save_model("tabular_model")
