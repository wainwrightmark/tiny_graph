pub (crate) const fn position_max(arr: &[u8]) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }

    let mut max = arr[0];
    let mut max_index = 0;
    let mut index = 1;
    while index < arr.len() {
        let item = arr[index];
        if item > max {
            max = item;
            max_index = index;
        }
        index += 1;
    }

    Some(max_index)
}

///An array `[0, 1, 2, .. N-1]`
pub (crate) const fn indexes_array<const N: usize>()->[u8;N]{
    let mut arr = [0;N];

    let mut index = 0usize;
    while index < arr.len() {
        arr[index] = index as u8;
        index+= 1;
    }


    arr
}