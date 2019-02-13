import re

# Grab missing (k, R) rules
missing = []
lines = open('missing').readlines()
for line in lines:
    mobj = re.match('Missing \S+ 1_xk(\d+)_(\S+)', line)
    missing.append((int(mobj.group(1)), float(mobj.group(2))))
missing = list(sorted(set(missing)))
# print missing

# Grab/sieve proposed rules
rules = {}
lines = open('rules').readlines()
count = 0
for line in lines:
    mobj = re.match('^\s*\{(\S+), \{(.*)\}\},\s*$', line)
    R = float(mobj.group(1))
    grids = []
    toks = re.split('\}, \{', mobj.group(2)[1:-1])
    for tok in toks:
        mobj = re.match(r'"(\S+)", "(\S+)", (\S+)', tok) 
        R2 = float(mobj.group(1))
        k = int(mobj.group(2))
        E = float(mobj.group(3))
        if R2 != R: raise RuntimeError("R2 != R")
        if (k, R2) in missing:
            # print (k, R2)
            count += 1
            continue
        grids.append((R2,k,E))
    rules[R] = grids
(count == len(missing))

# Print out rules
s = ''
for key in sorted(rules.keys()):
    rule = rules[key]
    s += '  {%.0E, {' % key
    rule2 = list(sorted(rule, key=lambda x:x[2]))
    for tind, tok in enumerate(rule2):
        R = tok[0]
        k = tok[1]
        E = tok[2]
        Rstr = '%.0E' % R
        mobj = re.match('(\d+)E\+(\d+)', Rstr)
        mant = int(mobj.group(1))
        exp = int(mobj.group(2))
        s += '{"%dE%d", "%02d", %.3E}' % (mant, exp, k, E)
        if tind + 1 != len(rule2):
            s += ', '
    s += '}},\n'
print s

        
