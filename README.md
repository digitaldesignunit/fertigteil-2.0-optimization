# fertigteil-2.0-optimization

[![Paper DOI](https://img.shields.io/badge/Paper_DOI-10.1007/978--3--031--13249--0_35-blue.svg)](http://dx.doi.org/10.1007/978-3-031-13249-0_35)
![Python](https://img.shields.io/badge/python-3.8-blue.svg)

This is the code for the matchmaking and environmental impact optimization done
within the *Fertigteil 2.0 (Precast Concrete Components 2.0)* research project.

# PLEASE NOTE THAT THIS CODE IS MOSTLY OLD AND WILL NOT BE WORKED ON ANYMORE

Hops Version: 0.16.2
    - rhinoinside>=0.6.0
    - ghhops-server>=1.5.5
    - rhino3dm>=8.17.0

## Table of Contents

[General](#general)  
[Installation & Updates](#installation--updates)  
[Development](#development)  
[Credits, Licensing & References](#credits) 


# General

## Malt

Malt is a collection of Hops components for Rhino Grasshopper.

- The Hops components run using a local [ghhops-server](https://github.com/mcneel/compute.rhino3d/tree/master/src/ghhops-server-py).
- The components are written in Python 3.8 and defined in `componentserver.py`.
- Rhino functionality is provided using [Rhino.Inside.Cpython](https://github.com/mcneel/rhino.inside-cpython).

## About the provided Conda Environment

The provided environment file `ft20-opt.yml` unifies the tools that are needed
to run the provided customized `componentserver.py` file.

## Prerequisites

If you want to use the provided Hops Components by running the componentserver locally on your machine, you need the following:
- Windows (unfortunately Hops and Rhino.Inside.Cpython won't work under OSX for now)
- [Anaconda / Miniconda](https://www.anaconda.com/products/individual)
- Rhino 7.4 or newer
- [Hops](https://developer.rhino3d.com/guides/grasshopper/hops-component/) ([Install using Rhino package manager by using the `_PackageManager` command](rhino://package/search?name=hops))

Currently, Malt is being tested to work using the following stack:
- Rhino 7 SR24 (7.24.22308.15001)
- Hops 0.15.3
- ghhops-server 1.5.3
- rhinoinside 0.6.0

While different Rhino and/or Hops versions *might* work, there is no guarantee
at the moment as malt is in a very early stage.

# Installation & Updates

## 1. Clone the repository into a directory of your choice

First off, clone or download this repository and unzip it (if needed) into a
working directory of your choice. For the purpose of this guide, I will use
`C:\source\repos` as my directory for all repositories. **If you have no idea
how to work with git or have never worked with a git repository before, please
have a look [here](docs/howto_git.md) first!**

If you want to clone using the Command line, `cd` into your repo directory, i.e.:
```
cd "C:\source\repos"
```

You can then clone the repository into your current working directory:
```
git clone https://github.com/digitaldesignunit/fertigteil-2.0-optimization.git
```

You should now end up with a new folder `fertigteil-2.0-optimization` inside
your working directory, containing all the files of the repository, so that the
full path is `C:\source\repos\fertigteil-2.0-optimization`

## 2. Set up the Virtual Environment using conda environment file

***NOTE: If you have not installed 
[Anaconda / Miniconda](https://www.anaconda.com/products/individual) yet, NOW
is the time to do it. If you have no idea how to get started with Anaconda,
please have a look [here](docs/howto_anaconda.md)***

Using a Windows Powershell, `cd` into the directory where **you** have 
cloned/unpacked the `fertigteil-2.0-optimization` repository.
For me that's running:
```
cd "C:\source\repos\fertigteil-2.0-optimization"
```

Set up a new conda virtual environment with the name `ft20-opt` using
the provided environment file by running:
```
conda env create -f "ft20-opt.yml"
```
*NOTE: This step can take quite some time and will take up a good amount of
space on your disk. You can expect at least ~5GB!*

Now we activate our newly created conda environment:
```
conda activate ft20-opt
```

## 3. Installing the malt python package

With the virtual environment activated and while in the root directory of the
fertigteil-2.0-optimization repository (where `setup.py` is located!), run the
following command:
```
pip install -e .
```

*NOTE: This will install the `malt` package and its submodules in **development
mode (recommended!)**, in case you want to extend and/or modify it. If you 
simply want to use the provided functions and components, you can also simply 
call `pip install .`*

## 4. Running the Hops Server in the Virtual Environment

Make sure your current working directory is the directory where `componentserver.py` 
is located. Otherwise browse to this directory using `cd` (as we did in step 3).
 Make sure the `ft20-opt` conda environment is active, otherwise run:
```
conda activate ft20-opt
```

Now you can start the Hops Server by running:
```
python componentserver.py
```

Note that you can also run the componentserver using different command line options:
- `python componentserver.py -d` will run the server in debug mode.
- `python componentserver.py -f` will run the server without using Flask.
- `python componentserver.py -n` will run the server in network access mode. **WARNING: THIS IS POTENTIALLY *VERY* DANGEROUS!**

## 5. Using one of the provided Hops Components in Grasshopper

Once the server is running, you can query it at different endpoints. When you
start the server, all available endpoints are printed to the console:

![Available Endpoints](/resources/readme/readme_01.png)

For a demo you can open the latest example file available in the `gh_dev`
folder. But you can of course also start from scratch:

Open Rhino and Grasshopper and start by placing a Hops Component on the canvas:

![Placing a new Hops component](/resources/readme/readme_02.png)

Doubleclick the Hops Component and set it to one of the available endpoints.
Note that the Hops Server is running under `http://localhost:5000/`.

![Setting an Endpoint](/resources/readme/readme_03.png)

The component that is available at this endpoint will then be loaded:

![Setting an Endpoint](/resources/readme/readme_04.png)

I recommend to run the available Hops Components asynchronously because this
will add a lot of responsiveness to your Grasshopper definition. I did not test
the caching functionality extensively, so feel free to experiment with that.
For more info on the available settings, please see [here](https://developer.rhino3d.com/guides/grasshopper/hops-component/#component-settings).

![Asynchronous execution](/resources/readme/readme_05.png)

You can now use the loaded Hops Component like any other Grasshopper component.
In this example, I first computed geodesic distances on a mesh from a source
vertex using the Heat Method available at the `/intri.HeatMethodDistance`
endpoint. Then I use the resulting values at each vertex to draw isocurves
on the mesh using the `/igl.MeshIsoCurves` endpoint.

![Geodesic Heat Isocurves](/resources/readme/readme_06.png)

## 6. Updating

### 6.1 Updating the `fertigteil-2.0-optimization` repository

To update your local repository, open a Powershell or Terminal and `cd` into
*your* directory of the repository, for me that's
```
cd "C:\source\repos\fertigteil-2.0-optimization"
```

Then you can update the repository using git:
```
git pull
```

If you have installed `malt` in development mode (see section 3) you`re already
done! If not, you have to install the updated module again. First activate
the conda virtual environment...
```
conda activate ddu_ias_research
```
...and then update the `malt` package by running
```
pip install .
```
...or update with installing in development mode this time by running
```
pip install -e .
```

### 6.2 Updating the conda environment

If you need to update your conda environment after the release of a new version
of the supplied `ddu_ias_research.yml` file, here is how you can do this:

First, `cd` into *your* `fertigteil-2.0-optimization` repository directory as always, for me that's
```
cd "C:\source\repos\fertigteil-2.0-optimization"
```

then update your conda environment by running
```
conda env update --name ft20-opt --file ft20-opt.yml --prune
```

*et voila* - your conda environment should now be updated with the newly
specified dependencies.

# Credits

## Citing

Please use the following publication for citations

```
@inproceedings{eschenbach_matter_2023,
	address = {Cham},
	title = {Matter as {Met}: {Towards} a {Computational} {Workflow} for {Architectural} {Design} with {Reused} {Concrete} {Components}},
	copyright = {All rights reserved},
	isbn = {978-3-031-13249-0},
	doi = {https://doi.org/10.1007/978-3-031-13249-0_35},
	abstract = {Over the past decades computational design, digital fabrication and optimisation have become widely adapted in architectural research, contemporary practice as well as in the construction industry. Nevertheless, current design and fabrication process-chains are still stuck in a linear notion of material use: building components are digitally designed, engineered, and ultimately materialised by consumption of raw materials. These can be defined as digital-real process chains. But these parametric design logics based on mass customisation inhibit the reuse of building components. In contrast to these predominant and established digital-real process chains, we propose a real-digital process chain: departing from our real, already materialised built environment. We digitise and catalogue physical concrete components within a component repository for future reuse. Subsequently, these components are reconditioned, enhanced if necessary and transitioned into a modular building system. The modularised components are then recombined to form a new building design and, eventually, a new building by combinatorial optimisation using mixed-integer linear programming (MILP). An accompanying life cycle assessment (LCA) complements the process and quantifies the environmental potential of reused building components. The paper presents research towards a feasible workflow for the reuse of structural concrete components. Furthermore, we suggest a digital repository, storing geometric as well as complementary data on the origin, history, and performances of the components to be reused. Here, we identify core data to be integrated in such a component repository.},
	booktitle = {Towards {Radical} {Regeneration}},
	publisher = {Springer International Publishing},
	author = {Eschenbach, Max Benjamin and Wagner, Anne-Kristin and Ledderose, Lukas and Böhret, Tobias and Wohlfeld, Denis and Gille-Sepehri, Marc and Kuhn, Christoph and Kloft, Harald and Tessmann, Oliver},
	editor = {Gengnagel, Christoph and Baverel, Olivier and Betti, Giovanni and Popescu, Mariana and Thomsen, Mette Ramsgaard and Wurm, Jan},
	year = {2023},
	pages = {442--455},
}
```

## Public Funding

This research was conducted within the Project _Fertigteil 2.0 -
Real-digital process chains for the production of built-in concrete
components_. The project [_Fertigteil 2.0 (Precast Concrete Components 2.0)_](https://www.remin-kreislaufwirtschaft.de/en/projects/fertigteil-20)
was funded by the Federal Ministry of Education and Research Germany (BMBF)
through the funding measure [Resource-efficient circular economy - Building and mineral cycles (ReMin)](https://www.fona.de/de/massnahmen/foerdermassnahmen/ressourceneffiziente-kreislaufwirtschaft-bauen-und-mineralische-stoffkreislaufe.php).

## Licensing & References

- Original code is licensed under the MIT License.
