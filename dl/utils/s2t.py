

# Multi source domain to single target domain
def ms2st(all, targets):
    s2ts = []
    for t in targets:
        s2ts.append({
                't': t,
                's': [_ for _ in all if _ != t]})
    return s2ts

def ss2st(sources, targets):
    s2ts = []
    for s in sources:
        for t in targets:
            if s==t:
                continue
            s2ts.append({
                    't': t,
                    's': [s]})
    return s2ts

class LazyMan2:
    """
    Input, domain_list, target_domain_list,
    for each target domain in target_domain_list,
    source domains are domain_list - target domain

    Multi source domain to single target domain
    """

    def __init__(self, ):
        self.all = all
        self.targets = targets
        # self.number_of_domains = len(domain_list)
        self.dfps = self.domain_transfer_pairs()

    def domain_transfer_pairs(self):
        dfps = []
        for t in self.targets:
            s = []
            for i in self.all:
                if i != t:
                    s.append(i)
            domain_dictionary = {
                'target_domain': t,
                's':s
            }
            source_and_target_domain_permutation_list.append(domain_dictionary)
        return source_and_target_domain_permutation_list


class LazyMan:
    """
    Input, domain_list, target_domain_list,
    for each target domain in target_domain_list,
    source domain can be any domain in domain_list - target domain

    Single source domain to single target domain
    """

    def __init__(self, domain_list, target_domain_list):
        self.domain_list = domain_list
        self.target_domain_list = target_domain_list
        # self.number_of_domains = len(domain_list)
        self.source_and_target_domain_permutation_list = \
            self._get_source_and_target_domain_permutation_list()

    def _get_source_and_target_domain_permutation_list(self):
        source_and_target_domain_permutation_list = []
        for target_domain in self.target_domain_list:
            for i in self.domain_list:
                if i != target_domain:
                    domain_dictionary = {
                        'target_domain': target_domain,
                        'source_domain': [i]
                    }
                    source_and_target_domain_permutation_list.append(domain_dictionary)
        return source_and_target_domain_permutation_list

#
# lazy_man = LazyMan(['CALTECH', 'LABELME', 'PASCAL', 'SUN'], ['LABELME','PASCAL', 'SUN'])
# for item in lazy_man.source_and_target_domain_permutation_list:
#     print(item['target_domain'], item['source_domain'])


##


# img = Image.open('/home/giorgio/Files/pycharm_project/fmc/data/VLCS/SUN/crossval/0/crossval_imgs_1.jpg')
# img.show()

