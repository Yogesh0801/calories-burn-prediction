// static/predictions/js/scripts.js
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    form.addEventListener('submit', function(event) {
        const inputs = form.querySelectorAll('input');
        let valid = true;
        inputs.forEach(input => {
            if (input.value === '') {
                valid = false;
                input.style.border = '2px solid red';
            } else {
                input.style.border = '';
            }
        });
        if (!valid) {
            event.preventDefault();
            alert('Please fill in all fields');
        }
    });
});
