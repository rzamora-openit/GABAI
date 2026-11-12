import streamlit as st

from Utility.approval_context import ApprovalContext

st.title("Approvals")


def decide(id, **decision):
    db = ApprovalContext()
    try:
        db.update(id, **decision)
    finally:
        db.close()


db = ApprovalContext()
try:
    approvals = db.getall()
finally:
    db.close()

if not approvals:
    st.info("No requests yet.")
else:
    # pending first (so they are the ones you see), then decided; oldest first within each group
    approvals.sort(key=lambda a: (a.approved or a.rejected, a.id))

    widths = [1, 2, 6, 2, 2, 2]
    for col, label in zip(st.columns(widths), ["ID", "Requested", "Request", "Status", "", ""]):
        col.markdown(f"**{label}**")

    for a in approvals:
        id_col, date_col, request_col, status_col, approve_col, reject_col = st.columns(widths, vertical_alignment="center")
        id_col.write(a.id)
        date_col.write(str(a.date_requested))
        request_col.write(a.request)
        if a.approved:
            status_col.markdown(":green[Approved]")
        elif a.rejected:
            status_col.markdown(":red[Rejected]")
        else:
            status_col.markdown(":orange[Pending]")
        if approve_col.button("Approve", key=f"approve_{a.id}", disabled=a.approved):
            decide(a.id, approved=True)
            st.rerun()
        if reject_col.button("Reject", key=f"reject_{a.id}", disabled=a.rejected):
            decide(a.id, rejected=True)
            st.rerun()
