# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import argparse
import clr
import logging
import json

# COMMAND LINE ARGUMENT PARSING -----------------------------------------------

# Create argument parser
arg_parser = argparse.ArgumentParser(description="Process arguments for MALT "
                                                 "component server.")
# Create arguments
arg_parser.add_argument("-d", "--debug",
                        action="store_true",
                        required=False,
                        help="Activates Flask debug mode. "
                             "Defaults to False.",
                        dest="debug")
arg_parser.add_argument("-n", "--networkaccess",
                        action="store_true",
                        required=False,
                        help="Activates network access mode. "
                             "Defaults to False.",
                        dest="networkaccess")
arg_parser.add_argument("-f", "--noflask",
                        action="store_false",
                        required=False,
                        help="Runs server using Hops standard HTTP server. "
                             "Defaults to False (uses Flask as middleware).",
                        dest="flask")
# Parse all command line arguments
cl_args = arg_parser.parse_args()


# OPTIONS ---------------------------------------------------------------------

# Make matplotlib logger less verbose to prevent imports in
# referenced libraries from triggering a wall of debug messages.
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Set to True to run in debug mode.
_DEBUG = cl_args.debug

# Set to True to allow access via local network (only works with Flask app!)
# WARNING: THIS MIGHT BE A SECURITY RISK BECAUSE IT POTENTIALLY ALLOWS PEOPLE
# TO EXECUTE CODE ON YOUR MACHINE! ONLY USE THIS IN A TRUSTED NETWORK!
_NETWORK_ACCESS = cl_args.networkaccess

# Set to True to run using Flask as middleware
_FLASK = cl_args.flask

# True if you want to run using Rhino.Inside.CPython
_RHINOINSIDE = True

# Set to True to enable System import
_USING_SYSTEM = True

# Set to True to enable Grasshopper import
_USING_GH = False

# Set to True to enable Kangaroo2 import
_USING_K2 = False

# HOPS & RHINO SETUP ----------------------------------------------------------

import ghhops_server as hs # NOQA402


# Define a custom Hops class to enable Rhino.Inside.CPython in
# combination with a Flask app (otherwise not possible)
class ExtendedHops(hs.Hops):
    """
    Custom extended Hops class allowing Flask app to also run Rhino.Inside.
    """

    def __new__(cls,
                app=None,
                debug=False,
                force_rhinoinside=False,
                *args,
                **kwargs) -> hs.base.HopsBase:
        # set logger level
        hs.hlogger.setLevel(hs.logging.DEBUG if debug else hs.logging.INFO)

        # determine the correct middleware base on the source app being wrapped
        # when running standalone with no source apps
        if app is None:
            hs.hlogger.debug("Using Hops default http server")
            hs.params._init_rhino3dm()
            return hs.middlewares.HopsDefault()

        # if wrapping another app
        app_type = repr(app)
        # if app is Flask
        if app_type.startswith("<Flask"):
            if force_rhinoinside:
                hs.hlogger.debug("Using Hops Flask middleware and rhinoinside")
                hs.params._init_rhinoinside()
            else:
                hs.hlogger.debug("Using Hops Flask middleware and rhino3dm")
                hs.params._init_rhino3dm()
            return hs.middlewares.HopsFlask(app, *args, **kwargs)

        # if wrapping rhinoinside
        elif app_type.startswith("<module 'rhinoinside'"):
            # determine if running with rhino.inside.cpython
            # and init the param module accordingly
            if not ExtendedHops.is_inside():
                raise Exception("rhinoinside is not loaded yet")
            hs.hlogger.debug("Using Hops default http server with rhinoinside")
            hs.params._init_rhinoinside()
            return hs.middlewares.HopsDefault(*args, **kwargs)

        raise Exception("Unsupported app!")


print("-----------------------------------------------------")
print("[INFO] Hops Server Configuration:")
print("[INFO] SERVER:  {0}".format(
            "Flask App" if _FLASK else "Hops Default HTTP Server"))
print("[INFO] RHINO:   {0}".format(
            "Rhino.Inside.CPython" if _RHINOINSIDE else "rhino3dm"))
if _NETWORK_ACCESS:
    print("[INFO] NETWORK: Network Access Enabled!")
    print("[WARNING] Enabling network access is a security risk because \n"
          "it potentially allows people to execute python code on your \n"
          "machine! Only use this option in a trusted network/environment!")
else:
    print("[INFO] NETWORK: Localhost Only")
print("-----------------------------------------------------")

# RHINO.INSIDE OR RHINO3DM
if _RHINOINSIDE:
    print("[INFO] Loading Rhino.Inside.CPython ...")
    import rhinoinside
    rhinoinside.load()
    import Rhino # NOQA402
else:
    import rhino3dm # NOQA402

# SYSTEM IF NECESSARY
if _USING_SYSTEM:
    print("[INFO] Loading System (.NET) ...")
    import System # NOQA402

# GRASSHOPPER IF NECESSARY
if _USING_GH:
    print("[INFO] Loading Grasshopper ...")
    clr.AddReference("Grasshopper.dll")
    import Grasshopper as gh # NOQA402

# KANGAROO 2 IF NECESSARY
if _USING_K2:
    print("[INFO] Loading Kangaroo2 ...")
    clr.AddReference("KangarooSolver.dll")
    import KangarooSolver as ks # NOQA402


# THIRD PARTY MODULE IMPORTS --------------------------------------------------

import numpy as np # NOQA402
from sklearn.manifold import TSNE # NOQA402
from sklearn.decomposition import PCA # NOQA402

# LOCAL MODULE IMPORTS --------------------------------------------------------

import malt # NOQA402
from malt import ft20 # NOQA402
from malt import hopsutilities as hsutil # NOQA402
from malt import miphopper # NOQA402


# REGSISTER FLASK AND/OR RHINOINSIDE HOPS APP ---------------------------------

if _FLASK:
    from flask import Flask # NOQA402
    flaskapp = Flask(__name__)
    hops = ExtendedHops(app=flaskapp, force_rhinoinside=_RHINOINSIDE)
elif not _FLASK and _RHINOINSIDE:
    hops = ExtendedHops(app=rhinoinside)
else:
    hops = ExtendedHops()


# HOPS COMPONENTS -------------------------------------------------------------

# GET ALL AVAILABLE COMPONENTS ////////////////////////////////////////////////

@hops.component(
    "/hops.AvailableComponents",
    name="AvailableComponents",
    nickname="Components",
    description="List all routes (URI's) of the available components",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[],
    outputs=[
        hs.HopsString("Components", "C", "All available Hops Components on this server.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsString("Description", "D", "The descriptions of the components", hs.HopsParamAccess.LIST), # NOQA501
    ])
def hops_AvailableComponentsComponent():
    comps = []
    descr = []
    for c in hops._components:
        uri = str(c)
        if not uri.startswith("/test."):
            comps.append(uri)
            descr.append(hops._components[c].description)
    return comps, descr


# GUROBI INTERFACE COMPONENTS /////////////////////////////////////////////////

@hops.component(
    "/gurobi.SolveAssignment2DPoints",
    name="SolveAssignment2DPoints",
    nickname="SolveAssignment2DPoints",
    description="Solve a 2d assignment problem given the datapoints using Gurobi.", # NOQA502
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsNumber("Design", "D", "The datapoints that define the design as DataTree of Numbers, where each Branch represents one Point.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsNumber("Inventory", "I", "The datapoints that define the inventory from which to choose the assignment as DataTree of Numbers, where each Branch represents one Point.", hs.HopsParamAccess.TREE), # NOQA501
    ],
    outputs=[
        hs.HopsInteger("Assignment", "A", "An optimal solution for the given assignment problem.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsNumber("Cost", "C", "The cost values for the optimal solution.", hs.HopsParamAccess.TREE), # NOQA501
    ])
def gurobi_SolveAssignment2DPointsComponent(design,
                                            inventory):

    # loop over trees and extract data points as numpy arrays
    design_p, np_design = hsutil.hops_tree_to_np_array(design)
    inventory_p, np_inventory = hsutil.hops_tree_to_np_array(inventory)

    # verify feasibility of input datapoints
    if np_design.shape[0] > np_inventory.shape[0]:
        raise ValueError("Number of Design datapoints needs to be smaller " +
                         "than or equal to number of Inventory datapoints!")

    # compute cost matrix
    cost = np.zeros((np_design.shape[0], np_inventory.shape[0]))
    for i, pt1 in enumerate(np_design):
        for j, pt2 in enumerate(np_inventory):
            cost[i, j] = np.linalg.norm(pt2 - pt1, ord=2)

    # solve the assignment problem using the gurobi interface
    assignment, assignment_cost = miphopper.solve_assignment_2d(cost)

    # return data as hops tree
    return (hsutil.np_int_array_to_hops_tree(assignment, design_p),
            hsutil.np_float_array_to_hops_tree(assignment_cost, design_p))


@hops.component(
    "/gurobi.SolveAssignment3DPoints",
    name="SolveAssignment3DPoints",
    nickname="SolveAssignment3DPoints",
    description="Solve a 3d assignment problem given the datapoints using Gurobi.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsNumber("Design", "D", "The datapoints that define the design as DataTree of Numbers, where each Branch represents one Point.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsNumber("Inventory", "I", "The datapoints that define the inventory from which to choose the assignment as DataTree of Numbers, where each Branch represents one Point.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsBoolean("SimplifyCase", "S", "Simplify the 3d problem case (or at least try to) by pre-computing the minimum cost and solving the resulting 2d cost matrix.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsInteger("Assignment", "A", "An optimal solution for the given assignment problem.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsNumber("Cost", "C", "The cost values for the optimal solution.", hs.HopsParamAccess.TREE), # NOQA501
    ])
def gurobi_SolveAssignment3DPointsComponent(design,
                                            inventory,
                                            simplify=False):

    # verify tree integrity
    if (not hsutil.hops_tree_verify(design) or
            not hsutil.hops_tree_verify(inventory)):
        raise ValueError("DataTree structure is inconsistent! All paths have "
                         "to be of the same shape!")

    # loop over design tree and extract data points as numpy arrays
    design_p, np_design = hsutil.hops_tree_to_np_array(design)

    # build inventory numpy array
    inventory_p, np_inventory = hsutil.hops_tree_to_np_array(inventory, True)
    inventory_shape = (len(set(p[0] for p in inventory_p)),
                       len(set(p[1] for p in inventory_p)),
                       len(np_inventory[0]))
    np_inventory_2d = np.zeros(inventory_shape)
    for path, data in zip(inventory_p, np_inventory):
        i = path[0]
        j = path[1]
        for k, d in enumerate(data):
            np_inventory_2d[i, j, k] = d

    # verify tree integrity
    if np_design.shape[0] > np_inventory_2d.shape[0]:
        raise ValueError("Number of Design datapoints needs to be smaller "
                         "than or equal to number of Inventory datapoints!")

    # simplifies the problem to a 2d assignment problem by pre-computing the
    # minimum cost and then solving a 2d assignment problem
    if simplify:
        # create empty 2d cost matrix
        cost = np.zeros((np_design.shape[0], inventory_shape[0]))
        mapping = np.zeros((np_design.shape[0], inventory_shape[0]),
                           dtype=int)

        # loop over all design objects
        for i, d_obj in enumerate(np_design):
            # loop over all objects in the inventory per design object
            for j in range(np_inventory_2d.shape[0]):
                # find minimum orientation and index of it
                pt1 = d_obj
                allcosts = [np.linalg.norm(np_inventory_2d[j, k] - pt1, ord=2)
                            for k in range(np_inventory_2d.shape[1])]
                mincost = min(allcosts)
                minidx = allcosts.index(mincost)
                # build cost matrix and store index in a mapping
                cost[i, j] = mincost
                mapping[i, j] = minidx

        # solve the assignment problem using the gurobi interface
        assignment, assignment_cost = miphopper.solve_assignment_2d(cost)

        assignment_3d = []
        for i, v in enumerate(assignment):
            assignment_3d.append((v, mapping[i, v]))
        assignment_3d = np.array(assignment_3d)

        # return data as hops tree
        return (hsutil.np_int_array_to_hops_tree(assignment_3d, design_p),
                hsutil.np_float_array_to_hops_tree(assignment_cost, design_p))
    else:
        # create empty 3d cost martix as np array
        cost = np.zeros((np_design.shape[0],
                         inventory_shape[0],
                         inventory_shape[1]))

        # loop over all design objects
        for i, d_obj in enumerate(np_design):
            # loop over all objects in the inventory per design object
            for j in range(np_inventory_2d.shape[0]):
                # loop over orientations for every object in the inventory
                for k in range(np_inventory_2d.shape[1]):
                    pt1 = d_obj
                    pt2 = np_inventory_2d[j, k]
                    cost[i, j, k] = np.linalg.norm(pt2 - pt1, ord=2)

        # solve the assignment problem using the gurobi interface
        assignment, assignment_cost = miphopper.solve_assignment_3d(cost)

    # return data as hops tree
    return (hsutil.np_int_array_to_hops_tree(assignment, design_p),
            hsutil.np_float_array_to_hops_tree(assignment_cost, design_p))


@hops.component(
    "/gurobi.SolveCuttingStockProblem",
    name="SolveCuttingStockProblem",
    nickname="SolveCSP",
    description="Solve a cutting stock problem.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsNumber("StockLength", "SL", "Stock Length", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("StockCrossSectionLong", "SCL", "Stock Cross Section Long Side", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("StockCrossSectionShort", "SCS", "Stock Cross Section Short Side", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("DemandLength", "DL", "Demand Length", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("DemandCrossSectionLong", "DCL", "Demand Cross Section Long Side", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("DemandCrossSectionShort", "DCS", "Demand Cross Section Short Side", hs.HopsParamAccess.LIST), # NOQA501
    ],
    outputs=[
        hs.HopsInteger("Assignment", "A", "An optimal solution for the given assignment problem.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("NewComponents", "N", "Components produced new.", hs.HopsParamAccess.TREE), # NOQA501
    ])
def gurobi_SolveCSPComponent(stock_len,
                             stock_cs_x,
                             stock_cs_y,
                             demand_len,
                             demand_cs_x,
                             demand_cs_y):

    # SANITIZE INPUT DATA -----------------------------------------------------

    if not len(stock_len) == len(stock_cs_x) == len(stock_cs_y):
        raise ValueError("Stock Length and Cross Section Size lists must "
                         "correspond in length!")
    if not len(demand_len) == len(demand_cs_x) == len(demand_cs_y):
        raise ValueError("Demand Length and Cross Section Size lists must "
                         "correspond in length!")

    # BUILD NP ARRAYS ---------------------------------------------------------

    m = np.column_stack((np.array([round(x, 6) for x in demand_len]),
                         np.array([round(x, 6) for x in demand_cs_x]),
                         np.array([round(x, 6) for x in demand_cs_y])))

    R = np.column_stack((np.array([round(x, 6) for x in stock_len]),
                         np.array([round(x, 6) for x in stock_cs_x]),
                         np.array([round(x, 6) for x in stock_cs_y])))

    # COMPOSE N ON BASIS OF M -------------------------------------------------

    cs_set = sorted(list(set([(x[1], x[2]) for x in m])), reverse=True)
    N = np.array([(float("inf"), x[0], x[1]) for x in cs_set])

    # RUN CUTTING STOCK OPTIMIZATION ------------------------------------------

    optimisation_result = miphopper.solve_csp(m, R, N)

    # RETURN THE OPTIMIZATION RESULTS -----------------------------------------

    return ([int(x[1]) for x in optimisation_result],
            hsutil.np_float_array_to_hops_tree(N))


# FERTIGTEIL 2.0 AND FARO API /////////////////////////////////////////////////

@hops.component(
    "/ft20.GetAllObjects",
    name="FT20GetAllObjects",
    nickname="FT20GetAllObjects",
    description="Get all objects from the FARO component repository server.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsBoolean("Refresh", "R", "Refresh the server.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsString("RepositoryComponents", "RC", "All repository components.", hs.HopsParamAccess.LIST), # NOQA501
    ])
def ft20_GetAllObjectsComponent(refresh: bool = True):

    stkey = 'FT20_COMPONENT_REPOSITORY'

    if stkey not in malt._STICKY.keys():
        malt._STICKY[stkey] = []
        refresh = True

    if refresh:
        components = ft20.api.get_all_objects()
        malt._STICKY[stkey] = components

    json_comps = [obj.JSON for obj in malt._STICKY[stkey]]
    return json_comps


@hops.component(
    "/ft20.CreateObject",
    name="FT20CreateObject",
    nickname="FT20CreateObject",
    description="Create a component object on the FARO component repository server.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsBoolean("Create", "C", "Create the objects on the server.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsString("RepositoryComponents", "R", "RepositoryComponents to create on the server.", hs.HopsParamAccess.LIST), # NOQA501
    ],
    outputs=[])
def ft20_CreateObjectComponent(create, repositorycomponents):
    if create:
        for i, comp in enumerate(repositorycomponents):
            cls_comp = ft20.RepositoryComponent.CreateFromJSON(comp)
            ft20.api.create_object(cls_comp)


@hops.component(
    "/ft20.ClearServer",
    name="FT20ClearServer",
    nickname="FT20ClearServer",
    description="Clear the FARO component repository server by deleting all components.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsBoolean("Clear", "C", "Clear the server.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[])
def ft20_ClearServerComponent(clear):
    if clear:
        all_components = ft20.api.get_all_objects()
        for comp in all_components:
            ft20.api.delete_object(comp.uid)


@hops.component(
    "/ft20.FT20OptimizeMatching",
    name="FT20OptimizeMatching",
    nickname="FT20Opt",
    description="Optimize a FT20 demand based on a given stock.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsString("RepositoryComponents", "RepositoryComponents", "Stock of components from the repository.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsString("DemandComponents", "DemandComponents", "Demand of components.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsString("ReUseCoeffs", "ReUseCoeffs", "ReUse Coefficients.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsString("ProductionCoeffs", "ProductionCoeffs", "Production Coefficients.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("CutLoss", "CutLoss", "Length Amount that is lost for each cut, i.e. through sawing.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("MIPGap", "MIPGap", "Acceptable MIPGap for Gurobi Solver.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsInteger("Assignment", "Assignment", "An optimal solution for the given demand.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("NewComponents", "NewProduction", "Components produced new.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsString("ResultObjects", "ResultObjects", "Components produced new.", hs.HopsParamAccess.LIST), # NOQA501
    ])
def ft20_FT20OptimizeMatchingComponent(repository_components,
                                       demand_components,
                                       reusecoeffs,
                                       productioncoeffs,
                                       cut_loss: float = 0.0,
                                       mipgap: float = 0.0):

    # SANITIZE JSON DATA AS CLASSES AND DICTS ---------------------------------

    repository_components = [
        ft20.RepositoryComponent.CreateFromDict(json.loads(obj))
        for obj in repository_components
        ]

    demand_components = [
        ft20.DemandComponent.CreateFromDict(json.loads(obj))
        for obj in demand_components
        ]

    reusecoeffs = json.loads(reusecoeffs)
    productioncoeffs = json.loads(productioncoeffs)

    # SANITIZE OTHER INPUT DATA -----------------------------------------------

    mipgap = abs(mipgap)
    mipgap = mipgap if mipgap <= 1.0 else 1.0

    # COMPUTE TRANSPORT DISTANCES ---------------------------------------------

    # for cS: - compute distance from origin location to nearest landfill
    #         - compute distance from site location to nearest concrete factory
    # NOTE - ASSUMPTION: site location is the same for all demand components!
    factory_distance = ft20.compute_factory_distance(
        demand_components[0].location)

    # for cM: compute transport from lab to site
    # NOTE - ASSUMPTION: site location is the same for all demand components!
    # NOTE - ASSUMPTION: current location is always a lab location!
    transport_to_site = ft20.compute_transport_to_site(
        repository_components,
        demand_components[0].location)

    # RUN FT2.0 MATCHING OPTIMIZATION -----------------------------------------

    optimization_result, N, result_objects = ft20.optimize_matching(
        repository_components,
        demand_components,
        cut_loss,
        factory_distance,
        transport_to_site,
        reusecoeffs,
        productioncoeffs,
        mipgap,
        verbose=True
    )

    # RETURN THE OPTIMIZATION RESULTS -----------------------------------------

    return ([int(x[1]) for x in optimization_result],
            hsutil.np_float_array_to_hops_tree(N),
            [json.dumps(ro) for ro in result_objects])


# SKLEARN /////////////////////////////////////////////////////////////////////

@hops.component(
    "/sklearn.TSNE",
    name="TSNE",
    nickname="TSNE",
    description="T-distributed Stochastic Neighbor Embedding.",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsNumber("Data", "D", "Point Data to be reduced using t-SNE as a DataTree, where each Branch represents one Point.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsInteger("Components", "N", "Dimension of the embedded space.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Perplexity", "P", "The perplexity is related to the number of nearest neighbors that are used in other manifold learning algorithms. Consider selecting a value between 5 and 50. Defaults to 30.", hs.HopsParamAccess.ITEM, ), # NOQA501
        hs.HopsNumber("EarlyExaggeration", "E", "Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. Defaults to 12.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("LearningRate", "R", "The learning rate for t-SNE is usually in the range (10.0, 1000.0). Defaults to 200.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Iterations", "I", "Maximum number of iterations for the optimization. Should be at least 250. Defaults to 1000.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Method", "M", "Barnes-Hut approximation (0) runs in O(NlogN) time. Exact method (1) will run on the slower, but exact, algorithm in O(N^2) time. Defaults to 0.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Initialization", "I", "Initialization method. Random (0) or PCA (1).", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("RandomSeed", "S", "Determines the random number generator. Pass an int for reproducible results across multiple function calls. Note that different initializations might result in different local minima of the cost function. Defaults to None.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("Points", "T", "The transformed points", hs.HopsParamAccess.TREE), # NOQA501
    ])
def sklearn_TSNEComponent(data,
                          n_components=2,
                          perplexity=30,
                          early_exaggeration=12.0,
                          learning_rate=200.0,
                          n_iter=1000,
                          method=0,
                          init=0,
                          rnd_seed=0):
    # loop over tree and extract data points
    paths, np_data = hsutil.hops_tree_to_np_array(data)
    # convert method string
    if method <= 0:
        method_str = "barnes_hut"
    else:
        method_str = "exact"
    if init <= 0:
        init_str = "random"
    else:
        init_str = "pca"
    # initialize t-SNE solver class
    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=rnd_seed,
                method=method_str,
                init=init_str)
    # run t-SNE solver on incoming data
    tsne_result = tsne.fit_transform(np_data)
    # return data as hops tree (dict)
    return hsutil.np_float_array_to_hops_tree(tsne_result, paths)


@hops.component(
    "/sklearn.PCA",
    name="PCA",
    nickname="PCA",
    description="Principal component analysis.",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsNumber("Data", "D", "Point Data to be reduced using PCA as a DataTree, where each Branch represents one Point.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsInteger("Components", "N", "Number of components (dimensions) to keep.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("Points", "T", "The transformed points", hs.HopsParamAccess.TREE), # NOQA501
    ])
def sklearn_PCAComponent(data,
                         n_components=2):
    # loop over tree and extract data points
    paths, np_data = hsutil.hops_tree_to_np_array(data)
    # initialize PCA solver class
    pca = PCA(n_components=n_components)
    # run PCA solver on incoming data
    pca_result = pca.fit_transform(np_data)
    # return data as hops tree (dict)
    return hsutil.np_float_array_to_hops_tree(pca_result, paths)


# RUN HOPS APP AS EITHER FLASK OR DEFAULT -------------------------------------

if __name__ == "__main__":
    print("-----------------------------------------------------")
    print("[INFO] Available Hops Components on this Server:")
    [print("{0} -> {1}".format(c, hops._components[c].description))
     for c in hops._components if not str(c).startswith("/test.")]
    print("-----------------------------------------------------")
    if type(hops) is hs.HopsFlask:
        if _NETWORK_ACCESS:
            flaskapp.run(debug=_DEBUG, host="0.0.0.0")
        else:
            flaskapp.run(debug=_DEBUG)
    else:
        hops.start(debug=_DEBUG)
