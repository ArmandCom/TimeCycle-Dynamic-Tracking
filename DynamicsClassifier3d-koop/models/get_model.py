# Note: Currently using Spatio-Temporal approach
from .DynClass import DynClass
# from .DynClass_spa-temp import DynClass

def get_model(opt):

  model = DynClass(opt)
  model.setup_training()
  model.initialize_weights()
  return model
