#![deny(unsafe_code)]

use franken_nlp::receipt::{CommitmentDomain, CommitmentKey, hmac_sha256};

#[test]
fn public_receipt_hmac_and_domain_commitments_are_distinct() {
    assert_eq!(
        hex(&hmac_sha256(&[0x0b; 20], b"Hi There")),
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
    );

    let key = CommitmentKey::new("receipt-key-fixture", [42; 32]).expect("valid public key id");
    let input = key
        .commit(
            CommitmentDomain::Input,
            "receipt-fixture",
            "input",
            b"private fixture",
        )
        .expect("input commitment");
    let output = key
        .commit(
            CommitmentDomain::Output,
            "receipt-fixture",
            "output",
            b"private fixture",
        )
        .expect("output commitment");
    let other_namespace = key
        .commit(
            CommitmentDomain::Input,
            "another-receipt",
            "input",
            b"private fixture",
        )
        .expect("namespace commitment");
    assert_ne!(
        input, output,
        "input and output commitments must remain domain separated"
    );
    assert_ne!(
        input, other_namespace,
        "receipt namespaces must remain domain separated"
    );
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(*byte >> 4)] as char);
        output.push(HEX[usize::from(*byte & 0x0f)] as char);
    }
    output
}
