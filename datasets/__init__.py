from .dataset import CustomDataSet, LightFieldTeacherDataSet


dataset_dict = {
    "all": CustomDataSet,
    "LightFieldTeacherDataSet": LightFieldTeacherDataSet,
}