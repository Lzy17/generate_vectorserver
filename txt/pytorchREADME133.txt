<p>This <code>./upstream</code> subfolder contains fixes for <code>FindCUDA</code> that are introduced in
later versions of cmake but cause generator expression errors in earlier CMake
versions. Specifically:</p>
<ol>
<li>
<p>a problem where a generator expression for include directories was
passed to NVCC, where the generator expression itself was prefixed by <code>-I</code>.
As the NNPACK include directory generator expression expands to multiple
directories, the second and later ones were not prefixed by <code>-I</code>, causing
NVCC to return an error. First fixed in CMake 3.7 (see
<a href="https://github.com/Kitware/CMake/commit/7ded655f">Kitware/CMake@7ded655f</a>).</p>
</li>
<li>
<p>Windows VS2017 fixes that allows one to define the ccbin path
differently between earlier versions of Visual Studio and VS2017. First
introduced after 3.10.1 master version (see
<a href="https://github.com/Kitware/CMake/commit/bc88329e">Kitware/CMake@bc88329e</a>).</p>
</li>
</ol>
<p>The downside of using these fixes is that <code>./upstream/CMakeInitializeConfigs.cmake</code>,
defining some new CMake variables (added in
<a href="https://github.com/Kitware/CMake/commit/48f7e2d3">Kitware/CMake@48f7e2d3</a>),
must be included before <code>./upstream/FindCUDA.cmake</code> to support older CMake
versions. A wrapper <code>./FindCUDA.cmake</code> is created to do this automatically, and
to allow submodules to use these fixes because we can't patch their
<code>CMakeList.txt</code>.</p>
<p>If you need to update files under <code>./upstream</code> folder, we recommend you issue PRs
against <a href="https://gitlab.kitware.com/cmake/cmake/tree/master/Modules/FindCUDA.cmake">the CMake mainline branch</a>,
and then backport it here for earlier CMake compatibility.</p>