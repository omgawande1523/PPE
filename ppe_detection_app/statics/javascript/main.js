async function fetchViolations() {
    try {
        const response = await fetch('/violations');
        const data = await response.json();

        const container = document.getElementById('violations-list');
        container.innerHTML = '';

        if (data.length === 0) {
            container.innerHTML = '<p>No violations detected 👷‍♂️</p>';
            return;
        }

        data.forEach(v => {
            const div = document.createElement('div');
            div.classList.add('violation-card');
            div.innerHTML = `
                <img src="data:image/jpeg;base64,${v.snapshot}" alt="Violation Snapshot">
                <p><strong>Person ID:</strong> ${v.person_id}</p>
                <p><strong>Missing:</strong> ${v.missing.join(', ')}</p>
            `;
            container.appendChild(div);
        });

    } catch (err) {
        console.error('Error fetching violations:', err);
    }
}

setInterval(fetchViolations, 3000); // Refresh every 3 seconds
