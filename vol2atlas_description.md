# vol2atlas — multi-modal volume alignment to the Allen Mouse Brain Atlas

vol2atlas is a software package developed to align large 3D imaging
datasets of mouse brain samples to the Allen Mouse Brain Reference
Atlas, the community-standard anatomical coordinate frame for mouse
neuroscience. Placing samples into a common atlas frame is a
prerequisite for cross-sample comparison, region-based quantification
(cell counting, signal averaging), and integration of results across
laboratories.

The package provides a complete workflow from raw 3D dataset to
atlas-registered output, organized as a sequence of command-line steps
that share a common project file. An initial alignment is performed
through interactive 3D and orthogonal-view tools — rough positioning
by translation and rotation sliders, fine refinement in standard
anatomical views, and optional landmark-based correction. A 3-letter
orientation code provided at project initialization seeds the rough
pose from the known scanner geometry, eliminating most manual rotation
in the interactive step. Landmarks are anchored to sample anatomy:
they remain valid across successive fits, allowing the user to
incrementally improve the registration by adding correspondences or
removing outliers without restarting from scratch. The atlas
resolution is interchangeable on the fly: the package supports the
Allen CCFv3 at 10, 25, 50, and 100 µm as well as alternative mouse
references (Kim, Osten, Princeton, Gubra, BlueBrain CCFv3-augmented),
and switching between Allen resolutions preserves all prior
registration work because coordinates are kept in physical units
throughout. Automated rigid and affine refinements based on mutual
information (via the ANTs library), including a joint optimization
that combines landmark constraints with intensity-based registration,
further reduce residual misalignment. All interactive operations run
in memory and only commit to disk on an explicit user save, so
exploratory fitting cannot corrupt prior work. The resulting
transformation is then applied to the full-resolution dataset,
producing standardized output files (NIfTI and multiscale OME-Zarr)
that integrate directly with downstream tools such as Neuroglancer
for inspection and BrainGlobe, ANTs, or elastix for deformable
refinement.

A central engineering requirement is scalability. Mouse brain micro-CT
datasets routinely reach tens to hundreds of gigabytes at full
resolution; the planned extension to electron microscopy will push
this into the terabyte range. To accommodate this, vol2atlas processes
data in small spatial blocks rather than loading whole volumes into
memory, with throughput distributed across multiple processing
threads.

**Status.** The micro-CT alignment pipeline is operational and has
been validated on real sample data, including partial-hemisphere
acquisitions for which off-the-shelf alternatives are not suitable.
The extension to electron microscopy — alignment of nanometer-
resolution EM datasets to the already-registered micro-CT volume — is
in the design phase, awaiting representative EM data for development
and testing.
