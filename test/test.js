import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  vus: 10,
  duration: '30s',
};

export default function () {
  const res = http.get(
    'https://aittorney.me/api/get_all_threads?user_id=user_2r7phtW8TyCt7i6U7RLs0DClelZ',
    {
      headers: {
        Cookie: '__clerk_db_jwt_YOUR_REAL_COOKIE_HERE',
      },
    }
  );

  check(res, {
    'Status 200': (r) => r.status === 200,
  });

  console.log('Response status:', res.status);
  sleep(1);
}
