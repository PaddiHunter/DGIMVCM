data_dict = {
    0: 'Scene-15',
    1: 'HW',
    2: 'LandUse-21',
    3: '100Leaves',
}
device='cuda:0'


def get_config(flag=1):
    """Determine the parameter information of the model"""
    data_name = data_dict[flag]
    if data_name in ['Scene-15']:
        return dict(
            dataset=data_name,
            topk=30,
            missing_rate=0.5,
            n_clustering=15,
            view_num=3,
            method='heat',
            hidden_size=1024,
            output_dim=128,
            view_dims=[20,59,40],
            n_layers=3,
            graph_rec_loss_weight=1.0,
            view_kl_loss_weight=1.0,
            graph_contrastive_loss_weight=1.0,
            training=dict(
                lr=1.0e-4,
                epoch=100,
                step_size = 25,
                data_seed=38,
            ),
        )
    elif data_name in ['HW']:
        return dict(
            dataset=data_name,
            topk=40,
            missing_rate=0.5,
            n_clustering=10,
            view_num=6,
            method='heat',
            hidden_size=1024,
            output_dim=256,
            view_dims=[240,76,216,47,64,6],
            n_layers=3,
            graph_rec_loss_weight=1.0,
            view_kl_loss_weight=1.0,
            graph_contrastive_loss_weight=1.0,
            training=dict(
                lr=1.0e-5,
                epoch=200,
                step_size=50,
                data_seed=29,
            )
        )


    elif data_name in ['LandUse-21']:
        return dict(
            dataset=data_name,
            topk=20,
            missing_rate=0.5,
            n_clustering=21,
            view_num=3,
            method='heat',
            hidden_size=1024,
            output_dim=128,
            view_dims=[20,59,40],
            n_layers=3,
            graph_rec_loss_weight=1.0,
            view_kl_loss_weight=1.0,
            graph_contrastive_loss_weight=1.0,
            training=dict(
                lr=1.0e-4,
                epoch=100,
                step_size = 25,
                data_seed=10,
            )
        )

    elif data_name in ['100Leaves']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.5,
            n_clustering=100,
            view_num=3,
            method='heat',
            hidden_size=1024,
            output_dim=256,
            n_layers=3,
            view_dims=[64,64,64],
            view_kl_loss_weight=1.0,
            graph_contrastive_loss_weight=1.0,
            graph_rec_loss_weight=1.0,
            training=dict(
                lr=1.0e-4,
                epoch=300,
                step_size = 10,
                data_seed=47,
            )
        )
