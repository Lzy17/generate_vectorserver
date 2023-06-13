<p>All the scripts in this directory are callable from <code>~/workspace/.circleci/scripts/foo.sh</code>.
Don't try to call them as <code>.circleci/scripts/foo.sh</code>, that won't
(necessarily) work.  See Note [Workspace for CircleCI scripts] in
job-specs-setup.yml for more details.</p>