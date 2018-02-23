from data import load_ted_data, split_dataset, TedDataset
from torch.utils.data import DataLoader
from model import MLP
from bonfire.trainer import Trainer
from bonfire.logger import BasicLogger
from sklearn.metrics.classification import accuracy_score


def run():

    # Config
    config = {
        'model_folder': 'tmp',
        'embedding_size': 50,
        'hidden_size': 25,
        'batch_size': 50,
        'epochs': 100
    }

    # Data
    tokens_ted, labels = load_ted_data('ted_en-20160408.xml')
    tokens_train, tokens_dev, tokens_test = split_dataset(tokens_ted)
    labels_train, labels_dev, labels_test = split_dataset(labels)

    train_dataset = TedDataset(tokens_train,
                               labels_train,
                               min_frequency=10)

    dev_dataset = TedDataset(tokens_dev,
                             labels_dev,
                             vocabulary=train_dataset.vocabulary,
                             raw_output=True)

    test_dataset = TedDataset(tokens_test,
                              labels_test,
                              vocabulary=train_dataset.vocabulary,
                              raw_output=True)

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=config['batch_size'],
        num_workers=4
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=config['batch_size'],
        num_workers=4
    )

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=config['batch_size'],
        num_workers=4
    )

    # Model
    model = MLP(config)
    model.initialize_features(data=train_dataset)
    model.build_model()

    # Logger
    logger = BasicLogger(metric=accuracy_score,
                         score_optimization='max')

    # Trainer
    trainer = Trainer(model=model, logger=logger)
    trainer.fit(train_dataloader, dev_dataloader, epochs=config['epochs'])

    model.load(
        '{}/{}.torch'.format(model.config['model_folder'],
                             type(model).__name__.lower())
    )

    target = []
    for batch in test_dataloader:
        target.extend(batch['output'].tolist())
    predictions = trainer.test(test_dataloader)
    print("Test Accuracy:", accuracy_score(target, predictions))


if __name__ == '__main__':
    run()
