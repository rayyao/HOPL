from .vot import VOTDataset, VOTLTDataset
from .otb import OTBDataset
from .uav import UAVDataset
from .lasot import LaSOTDataset
from .nfs import NFSDataset
from .trackingnet import TrackingNetDataset
from .got10k import GOT10kDataset

class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        #assert 'name' in kwargs, "VOT2019"
        name = kwargs['name']
        if 'OTB100' == name:
            dataset = OTBDataset(**kwargs)
        elif 'LaSOT' == name:
            dataset = LaSOTDataset(**kwargs)
        elif 'UAV' == name:
            dataset = UAVDataset(**kwargs)
        elif 'NFS' == name:
            dataset = NFSDataset(**kwargs)
        elif 'VOT2018' == name or 'VOT2016' == name or 'VOT2019' == name:
            dataset = VOTDataset(**kwargs)
        elif 'VOT2018-LT' == name:
            dataset = VOTLTDataset(**kwargs)
        elif 'TrackingNet' == name:
            dataset = TrackingNetDataset(**kwargs)
        elif 'GOT-10k' == name:
            dataset = GOT10kDataset(**kwargs)
        else:
            #dataset = otherDataset(**kwargs)
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset
