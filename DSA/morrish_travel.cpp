// morrish traversal for inorder BT
 
vector<int> getInorder(TreeNode* root){
    vector<int> inorder;
    TreeNode *cur = root;
    while(cur != Null){
        
        if(cur -> left == NULL){
            inorder.push_back(cur -> val);
            cur = cur -> right;
        }

        else {
            TreeNode *prev = cur -> left;
            while(prev -> right && prev -> right != cur){
                prev = prev -> right;
            }

            if(prev -> right ==NULL){
                prev -> right = cur;
                cur = cur -> left;
            }

            else {
                prev -> right = NULL;
                inorder.push_back(cur.val);
                cur = cur -> right;
            }
        }
    }
    return inorder;
}

//  for preorder

vector<int> getInorder(TreeNode* root){
    vector<int> inorder;
    TreeNode *cur = root;
    while(cur != Null){
        
        if(cur -> left == NULL){
            inorder.push_back(cur -> val);
            cur = cur -> right;
        }

        else {
            TreeNode *prev = cur -> left;
            while(prev -> right && prev -> right != cur){
                prev = prev -> right;
            }

            if(prev -> right ==NULL){
                prev -> right = cur;
                inorder.push_back(cur.val); // this line change
                cur = cur -> left;
            }

            else {
                prev -> right = NULL;
                // from here 
                cur = cur -> right;
            }
        }
    }
    return inorder;
}
